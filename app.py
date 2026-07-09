from __future__ import annotations

import re
import shutil
import sys
import textwrap
from collections.abc import Callable

from config import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_MODE,
    DEFAULT_RUNTIME_PREFERENCE,
    HISTORY_PREVIEW_LIMIT,
    MAX_DISPLAY_WIDTH,
    RECENT_CONTEXT_LIMIT,
    SUMMARY_HISTORY_LIMIT,
)
from core.chatbot import OllamaError
from core.image_generation import ImageGenerationError, generate_sonic_image
from core.live_web import sanitize_live_web_output
from core.language import LanguageContext, language_metadata, resolve_language_context
from core.memory import (
    clear_history,
    ensure_chat_history_file,
    ensure_saved_notes_file,
    get_recent_context,
    load_history,
    save_chat_record,
    save_message,
    save_note as persist_note,
)
from features.prompt_gen import build_prompt_generation_prompt

from core.providers import SonicProviderError, dispatch_sonic_prompt
from core.router import RouteResult, route_user_input

TASK_BUILDERS: dict[str, Callable[..., str]] = {
    "prompt": build_prompt_generation_prompt,
}

TASK_SYSTEM_MODES = {
    "prompt": "prompt",
}


class _TerminalStreamWriter:
    def __init__(self, flush_threshold: int = 80) -> None:
        self.received_any = False
        self._buffer: list[str] = []
        self._buffer_length = 0
        self._flush_threshold = max(1, flush_threshold)

    def _write(self, text: str) -> None:
        if not text:
            return

        sys.stdout.write(text)
        sys.stdout.flush()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        buffered_text = "".join(self._buffer)
        self._buffer.clear()
        self._buffer_length = 0
        self._write(buffered_text)

    def __call__(self, chunk: str) -> None:
        if not chunk:
            return

        self.received_any = True
        self._buffer.append(chunk)
        self._buffer_length += len(chunk)

        if "\n" in chunk or self._buffer_length >= self._flush_threshold:
            self._flush_buffer()

    def finish(self) -> None:
        self._flush_buffer()


def _terminal_width() -> int:
    width = shutil.get_terminal_size((MAX_DISPLAY_WIDTH, 20)).columns
    return max(72, min(MAX_DISPLAY_WIDTH, width))


def _section_line(char: str = "=") -> str:
    return char * _terminal_width()


def _clean_text_preview(text: str, limit: int = 96) -> str:
    cleaned_text = " ".join((text or "").split())
    if len(cleaned_text) <= limit:
        return cleaned_text
    return textwrap.shorten(cleaned_text, width=limit, placeholder="...")


def _parse_limit(argument: str, default: int) -> int:
    if not argument:
        return default

    match = re.search(r"\d+", argument)
    if not match:
        return default

    parsed_value = int(match.group(0))
    return max(1, min(parsed_value, 50))


def _strip_outer_code_fence(text: str) -> str | None:
    cleaned_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n?(.*)\n?```", cleaned_text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).replace("\r\n", "\n").replace("\r", "\n")


def _format_plain_terminal_text(text: str, width: int) -> str:
    paragraphs = [paragraph for paragraph in (text or "").strip().split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return ""

    formatted_blocks: list[str] = []
    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        if any(line.strip().startswith("```") for line in lines):
            formatted_blocks.append("\n".join(lines))
            continue

        if any(line.startswith(("    ", "\t")) for line in lines):
            formatted_blocks.append("\n".join(lines))
            continue

        if all(line.strip().startswith(("-", "*", ">", "•")) or re.match(r"^\d+\.", line.strip()) for line in lines if line.strip()):
            formatted_blocks.append("\n".join(line.rstrip() for line in lines))
            continue

        joined_text = " ".join(line.strip() for line in lines).strip()
        formatted_blocks.append(textwrap.fill(joined_text, width=width))

    return "\n\n".join(formatted_blocks)


def format_terminal_response(text: str, prefer_code: bool = False) -> str:
    cleaned_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned_text:
        return ""

    if prefer_code:
        stripped_code = _strip_outer_code_fence(cleaned_text)
        if stripped_code is not None:
            return stripped_code
        return cleaned_text

    return cleaned_text


def _format_runtime_label(runtime_preference: str) -> str:
    normalized_runtime = (runtime_preference or "").strip().lower()
    if normalized_runtime in {"online", "live", "liveweb", "live_web", "web"}:
        return "Online"
    return "Offline"


def print_banner() -> None:
    print(_section_line())
    print(f"{APP_NAME} - {APP_DESCRIPTION}")
    print(f"Runtime: {_format_runtime_label(DEFAULT_RUNTIME_PREFERENCE)}")
    print(
        "Commands: /prompt /image /history /clear /save-note /summarize help exit"
    )
    print(_section_line())
    print()


def build_help_text() -> str:
    return (
        "Commands:\n"
        "  /history [limit]\n"
        "  /clear\n"
        "  /save-note [text]\n"
        "  /summarize [limit]\n"
        "  /prompt <idea>\n"
        "  /image <prompt>\n"
        "  help\n"
        "  exit | quit\n"
    )


def print_help() -> None:
    print(build_help_text())


def print_history(entries: list[dict[str, object]]) -> None:
    if not entries:
        print("No chat history yet.")
        return

    print("Recent history:")
    print(_section_line("-"))
    for index, entry in enumerate(entries, start=1):
        timestamp = str(entry.get("timestamp", "")).strip() or "unknown"
        user_message = str(entry.get("user_message") or entry.get("user_input") or "").strip()
        assistant_response = str(entry.get("assistant_response") or entry.get("assistant_reply") or "").strip()

        print(f"{index}. [{timestamp}]")
        print(f"   You   : {_clean_text_preview(user_message, 120)}")
        print(f"   SONIC : {_clean_text_preview(assistant_response, 160)}")
        print()


def build_prompt_for_route(route: RouteResult, language_context: LanguageContext) -> tuple[str, str, str]:
    if route.action == "task":
        builder = TASK_BUILDERS.get(route.task_name)
        if builder is None:
            raise ValueError(f"Unsupported task '{route.task_name}'.")
        system_mode = TASK_SYSTEM_MODES.get(route.task_name, route.task_name or route.mode)
        return (
            system_mode,
            get_system_prompt(system_mode, language_context=language_context),
            builder(route.content, language_context=language_context),
        )

    if route.action == "summarize":
        recent_history = get_recent_context(limit=_parse_limit(route.content, SUMMARY_HISTORY_LIMIT))
        return (
            "summary",
            get_system_prompt("summary", language_context=language_context),
            build_summary_prompt(recent_history, language_context=language_context),
        )

    recent_history = get_recent_context(limit=RECENT_CONTEXT_LIMIT)
    return (
        "assistant",
        get_system_prompt("assistant", language_context=language_context),
        build_chat_prompt(route.content, recent_history, language_context=language_context),
    )


def handle_history(route: RouteResult) -> None:
    history_limit = _parse_limit(route.content, HISTORY_PREVIEW_LIMIT)
    print_history(load_history(limit=history_limit))


def handle_clear_history() -> None:
    cleared_count = clear_history()
    print(f"Cleared {cleared_count} history entries.")


def handle_save_note(route: RouteResult, last_assistant_reply: str, current_mode: str = "assistant") -> str:
    note_text = route.content.strip() if route.content.strip() else last_assistant_reply.strip()
    if not note_text:
        print("Error: No note text available to save.")
        return last_assistant_reply

    persist_note(note_text)
    confirmation = "Saved note to data/chats/saved_notes.txt."
    print(confirmation)
    save_message(
        mode=current_mode,
        user_message=route.raw_input,
        assistant_response=confirmation,
        route_type="save_note",
        task_name="note",
    )
    return confirmation


def handle_model_response(
    route: RouteResult,
) -> str:
    language_history = get_recent_context(limit=RECENT_CONTEXT_LIMIT)
    language_message = route.content if route.content else route.raw_input
    language_context = resolve_language_context(language_message, language_history)
    resolved_mode, system_prompt, prompt_text = build_prompt_for_route(route, language_context)
    options_mode = resolved_mode
    prefer_code = False
    stream_writer = _TerminalStreamWriter()
    live_web_query = route.content if route.action in {"chat", "task"} else ""
    live_web_history = get_recent_context(limit=RECENT_CONTEXT_LIMIT) if route.action == "chat" else []

    print()
    try:
        sonic_response = dispatch_sonic_prompt(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            mode=options_mode,
            runtime_preference=DEFAULT_RUNTIME_PREFERENCE,
            live_web_query=live_web_query,
            live_web_history=live_web_history,
            stream_callback=stream_writer,
            language_context=language_context,
        )
        reply = sonic_response["reply"]
        notice = str(sonic_response.get("notice") or "").strip()
        if str(sonic_response["runtime_used"]).lower() in {"online", "live_web"}:
            reply = sanitize_live_web_output(reply, query=live_web_query) or "I couldn't find fresh results for that query."
    finally:
        stream_writer.finish()

    if stream_writer.received_any:
        print()
    else:
        if notice:
            print(notice)
            print()
        print(format_terminal_response(reply, prefer_code=prefer_code))
        print()

    selected_mood = resolved_mode
    task_name = route.task_name if route.action == "task" else None
    if route.action == "summarize":
        task_name = "summary"

    save_chat_record(
        user_input=route.raw_input,
        assistant_reply=reply,
        selected_mood=selected_mood,
        route_type=route.action,
        task_name=task_name,
        effective_mode=resolved_mode,
        metadata={
            "runtime_used": sonic_response["runtime_used"],
            "provider_used": sonic_response["provider_used"],
            "model_used": sonic_response["model_used"],
            "requested_runtime": sonic_response.get("requested_runtime", DEFAULT_RUNTIME_PREFERENCE),
            "fallback_used": bool(sonic_response.get("fallback_used")),
            "fallback_reason": sonic_response.get("fallback_reason", ""),
            "notice": notice,
            **language_metadata(language_context),
        },
    )
    return reply


def handle_image_response(route: RouteResult) -> str:
    cleaned_prompt = route.content.strip()
    print()
    try:
        image_result = generate_sonic_image(cleaned_prompt)
    except (ImageGenerationError, ValueError) as exc:
        error_reply = f"Error: {exc}"
        print(error_reply)
        save_chat_record(
            user_input=route.raw_input,
            assistant_reply=error_reply,
            selected_mood="image",
            route_type="image",
            task_name="image",
            effective_mode="image",
            metadata={
                "provider_used": "error",
                "model_used": "-",
                "requested_runtime": "online",
                "fallback_used": False,
                "fallback_reason": "",
            },
        )
        return error_reply

    reply = f"Generated image for: {image_result.prompt}"
    print(reply)
    print(f"Saved to: {image_result.image_path}")
    print()

    save_chat_record(
        user_input=route.raw_input,
        assistant_reply=reply,
        selected_mood="image",
        route_type="image",
        task_name="image",
        effective_mode="image",
        metadata={
            "provider_used": image_result.provider_used,
            "model_used": image_result.model_used,
            "requested_runtime": "online",
            "fallback_used": False,
            "fallback_reason": "",
            "image_path": image_result.image_path,
            "image_url": image_result.image_url,
            "api_endpoint": image_result.api_endpoint,
        },
    )
    return reply


def main() -> None:
    ensure_chat_history_file()
    ensure_saved_notes_file()

    last_assistant_reply = ""
    print_banner()

    while True:
        try:
            user_input = input("SONIC > ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_input:
            continue

        route = route_user_input(user_input, "assistant")

        if route.action == "help":
            print_help()
            continue

        if route.action == "exit":
            print("Goodbye.")
            break

        if route.action == "error":
            print(f"Error: {route.message}")
            continue

        if route.action == "empty":
            continue

        if route.action == "history":
            handle_history(route)
            continue

        if route.action == "clear":
            handle_clear_history()
            last_assistant_reply = ""
            continue

        if route.action == "save_note":
            handle_save_note(route, last_assistant_reply, "assistant")
            continue

        if route.action in {"chat", "task", "summarize"}:
            try:
                last_assistant_reply = handle_model_response(route)
            except (OllamaError, SonicProviderError, ValueError) as exc:
                print(f"Error: {exc}")
            except KeyboardInterrupt:
                print()
                break
            continue

        if route.action == "image":
            try:
                last_assistant_reply = handle_image_response(route)
            except KeyboardInterrupt:
                print()
                break
            continue

        print("Error: Unknown route. Type help to see available commands.")


if __name__ == "__main__":
    main()
