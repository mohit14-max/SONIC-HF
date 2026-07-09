from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from textwrap import shorten

from config import RECENT_CONTEXT_LIMIT
from core.language import LanguageContext, build_language_instruction


SONIC_PERSONA_DIRECTIVE = (
    "You are SONIC, an offline-first AI workspace assistant. "
    "Always speak as SONIC. "
    "Never claim to be NVIDIA, Nemotron, Google, SerpAPI, or any hosted provider. "
    "If the user asks who you are, answer directly as SONIC."
)


def _sonic_prompt(body: str) -> str:
    return f"{SONIC_PERSONA_DIRECTIVE} {body}"


IDENTITY_LEAK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*(?:i am|i'm|i've|i have been).*(?:nemotron|nvidia|serpapi|google)\b",
        r"^\s*(?:created by|powered by)\s+(?:nvidia|nemotron|serpapi|google)\b",
        r"^\s*i am a large language model.*(?:nvidia|nemotron|serpapi)\b",
        r"^\s*hosted model\b",
    )
)


def _strip_identity_leaks(text: str) -> str:
    cleaned_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned_text.strip():
        return ""

    sanitized_lines: list[str] = []
    for raw_line in cleaned_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            sanitized_lines.append("")
            continue

        if any(pattern.search(line) for pattern in IDENTITY_LEAK_PATTERNS):
            continue

        sanitized_lines.append(line)

    while sanitized_lines and not sanitized_lines[0].strip():
        sanitized_lines.pop(0)
    while sanitized_lines and not sanitized_lines[-1].strip():
        sanitized_lines.pop()

    return "\n".join(sanitized_lines).strip()


sanitize_identity_leaks = _strip_identity_leaks


SONIC_SYSTEM_PROMPT = _sonic_prompt(
    "Be helpful, clear, precise, and conversational. Provide direct and useful answers, code, or explanations as needed."
)

PROMPT_GENERATOR_PROMPT = (
    f"{SONIC_PERSONA_DIRECTIVE} "
    "You are SONIC in PROMPT GENERATOR MODE. "
    "Create polished prompts for images, anime, cinematic scenes, or creative writing. "
    "Keep them specific, vivid, and production-ready."
)

SUMMARY_SYSTEM_PROMPT = (
    f"{SONIC_PERSONA_DIRECTIVE} "
    "You are SONIC in SUMMARY MODE. "
    "Summarize recent chat clearly and briefly. "
    "Focus on main topics, useful answers, open questions, and next steps."
)

SYSTEM_PROMPTS: dict[str, str] = {
    "prompt": PROMPT_GENERATOR_PROMPT,
    "prompt_gen": PROMPT_GENERATOR_PROMPT,
    "summary": SUMMARY_SYSTEM_PROMPT,
}


def get_system_prompt(mode: str | None = None, language_context: LanguageContext | None = None) -> str:
    normalized_mode = (mode or "").strip().lower().lstrip("/")
    system_prompt = SYSTEM_PROMPTS.get(normalized_mode, SONIC_SYSTEM_PROMPT)

    if language_context is None:
        return system_prompt

    return f"{system_prompt} {build_language_instruction(language_context)}"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _preview_text(text: str, limit: int = 200) -> str:
    cleaned_text = _normalize_text(_strip_identity_leaks(text))
    if len(cleaned_text) <= limit:
        return cleaned_text
    return shorten(cleaned_text, width=limit, placeholder="...")


def format_history_entries(
    entries: Sequence[Mapping[str, object]] | None,
    limit: int | None = RECENT_CONTEXT_LIMIT,
) -> str:
    if not entries:
        return ""

    selected_entries = list(entries)
    if limit is not None and limit > 0:
        selected_entries = selected_entries[-limit:]

    formatted_blocks: list[str] = []
    for index, entry in enumerate(selected_entries, start=1):
        timestamp = str(entry.get("timestamp", "")).strip() or "unknown"
        user_message = str(entry.get("user_message") or entry.get("user_input") or "").strip()
        assistant_response = _strip_identity_leaks(
            str(entry.get("assistant_response") or entry.get("assistant_reply") or "")
        ).strip()

        formatted_blocks.append(
            "\n".join(
                [
                    f"[{index}] {timestamp}",
                    f"User: {_preview_text(user_message)}",
                    f"SONIC: {_preview_text(assistant_response)}",
                ]
            )
        )

    return "\n\n".join(formatted_blocks)


def build_chat_prompt(
    user_message: str,
    recent_history: Sequence[Mapping[str, object]] | None = None,
    language_context: LanguageContext | None = None,
) -> str:
    cleaned_message = (user_message or "").strip()
    if not cleaned_message:
        raise ValueError("User message cannot be empty.")

    sections: list[str] = [
        "Identity: You are SONIC. Do not claim to be NVIDIA, Nemotron, Google, SerpAPI, or a hosted model.",
        "Reply to the latest user message directly.",
        "Use recent context only if needed.",
    ]
    if language_context is not None:
        sections.append(build_language_instruction(language_context))

    history_text = format_history_entries(recent_history, limit=RECENT_CONTEXT_LIMIT)
    if history_text:
        sections.append("Recent context:\n" + history_text)

    sections.append("Latest user message:\n" + cleaned_message)
    return "\n\n".join(sections)


def build_summary_prompt(
    recent_history: Sequence[Mapping[str, object]] | None,
    language_context: LanguageContext | None = None,
) -> str:
    history_text = format_history_entries(recent_history, limit=None)
    if not history_text:
        raise ValueError("No history available to summarize.")

    language_instruction = ""
    if language_context is not None:
        language_instruction = build_language_instruction(language_context) + "\n"

    return (
        "Identity: You are SONIC. Do not claim to be NVIDIA, Nemotron, Google, SerpAPI, or a hosted model.\n"
        f"{language_instruction}"
        "Summarize the recent SONIC conversation in short bullets.\n"
        "Include the main topics, useful answers, open questions, and next steps.\n"
        "Keep it brief and readable.\n\n"
        f"Conversation:\n{history_text}"
    )
