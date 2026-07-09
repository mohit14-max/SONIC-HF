from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from config import (
    DEFAULT_MODEL_NAME,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_URL,
    REQUEST_TIMEOUT_SECONDS,
    get_generation_options,
)


class OllamaError(RuntimeError):
    pass


CODE_FENCE_PATTERN = re.compile(r"```(?:([a-zA-Z0-9_+-]+))?\s*\n?(.*?)```", re.DOTALL)

INTRO_LINE_PREFIXES = (
    "here is the code",
    "here's the code",
    "here is the python code",
    "here's the python code",
    "here is your code",
    "here's your code",
    "here is the final code",
    "here's the final code",
    "here is the corrected code",
    "here's the corrected code",
    "here is a clean version",
    "here's a clean version",
    "here is a simple version",
    "here's a simple version",
    "below is the code",
    "below is your code",
    "sure, here's the code",
    "sure, here is the code",
    "sonic's",
    "the code is",
    "final code:",
    "code:",
    "here you go",
    "i wrote",
    "i have written",
    "i've written",
    "this should help",
)

OUTRO_LINE_PREFIXES = (
    "explanation:",
    "note:",
    "if you want",
    "let me know",
    "let me know if you want",
    "you can also",
    "feel free",
    "this code",
    "that should",
    "this should",
)


LONG_RESPONSE_HINTS = (
    "detailed",
    "complete",
    "full plan",
    "full explanation",
    "step by step",
    "step-by-step",
    "in detail",
    "thorough",
    "comprehensive",
    "deeper",
    "more detail",
    "elaborate",
)

BRIEF_RESPONSE_HINTS = (
    "brief",
    "concise",
    "short",
    "quick",
    "just the answer",
    "one line",
    "one sentence",
    "short answer",
    "tl;dr",
    "tldr",
)

LONG_RESPONSE_NUM_PREDICT_BY_MODE = {
    "prompt": 768,
    "summary": 384,
}

BRIEF_RESPONSE_NUM_PREDICT_BY_MODE = {
    "prompt": 240,
    "summary": 192,
}

REQUEST_SECTION_MARKERS = (
    "Latest user message:\n",
    "Task:\n",
    "Content:\n",
    "Topic:\n",
    "Idea:\n",
)


def _clean_response_text(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _extract_request_section(prompt: str) -> str:
    cleaned_prompt = prompt or ""
    latest_index = -1
    latest_marker = ""

    for marker in REQUEST_SECTION_MARKERS:
        index = cleaned_prompt.rfind(marker)
        if index > latest_index:
            latest_index = index
            latest_marker = marker

    if latest_index >= 0:
        return cleaned_prompt[latest_index + len(latest_marker) :]

    return cleaned_prompt


def _prompt_requests_long_answer(prompt: str) -> bool:
    normalized_prompt = _normalize_text(_extract_request_section(prompt))
    return _contains_any(normalized_prompt, LONG_RESPONSE_HINTS)


def _prompt_requests_brief_answer(prompt: str) -> bool:
    normalized_prompt = _normalize_text(_extract_request_section(prompt))
    return _contains_any(normalized_prompt, BRIEF_RESPONSE_HINTS)


def _resolve_num_predict(mode: str, prompt: str, current_num_predict: int) -> int:
    normalized_mode = (mode or "").strip().lower()
    if _prompt_requests_long_answer(prompt):
        long_floor = LONG_RESPONSE_NUM_PREDICT_BY_MODE.get(normalized_mode, 768)
        return max(current_num_predict, long_floor)

    if _prompt_requests_brief_answer(prompt):
        brief_ceiling = BRIEF_RESPONSE_NUM_PREDICT_BY_MODE.get(normalized_mode, 256)
        return min(current_num_predict, brief_ceiling)

    return current_num_predict


def _extract_stream_chunk_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.replace("\r\n", "\n").replace("\r", "\n")

    if not isinstance(payload, Mapping):
        return ""

    for key in ("response", "content", "text", "output", "answer", "completion"):
        candidate = payload.get(key)
        if isinstance(candidate, str):
            return candidate.replace("\r\n", "\n").replace("\r", "\n")

    message = payload.get("message")
    if isinstance(message, Mapping):
        message_text = message.get("content")
        if isinstance(message_text, str):
            return message_text.replace("\r\n", "\n").replace("\r", "\n")

    return ""





def _extract_response_text(payload: Any) -> str:
    if isinstance(payload, str):
        return _clean_response_text(payload)

    if isinstance(payload, Mapping):
        for key in ("response", "content", "text", "output", "answer", "completion", "message"):
            candidate = payload.get(key)
            if candidate is None:
                continue

            extracted_text = _extract_response_text(candidate)
            if extracted_text:
                return extracted_text

        choices = payload.get("choices")
        if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
            for choice in choices:
                extracted_text = _extract_response_text(choice)
                if extracted_text:
                    return extracted_text

        for key in ("data", "result", "results", "error"):
            candidate = payload.get(key)
            if candidate is None:
                continue

            extracted_text = _extract_response_text(candidate)
            if extracted_text:
                return extracted_text

        return ""

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            extracted_text = _extract_response_text(item)
            if extracted_text:
                return extracted_text

    return ""


def _normalize_mode(mode: str | None) -> str:
    return (mode or "").strip().lower().lstrip("/")


def _coerce_positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _request_ollama(payload: Mapping[str, Any], timeout: int | None = None) -> urllib_request.addinfourl:
    request_timeout = timeout if timeout is not None else REQUEST_TIMEOUT_SECONDS
    request_data = json.dumps(dict(payload)).encode("utf-8")
    request = urllib_request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        return urllib_request.urlopen(request, timeout=request_timeout)
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if detail:
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        raise OllamaError(f"Ollama returned HTTP {exc.code}.") from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc


def build_ollama_payload(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    options: Mapping[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty.")

    normalized_mode = _normalize_mode(mode)
    generation_options = get_generation_options(mode=normalized_mode, overrides=options)
    current_num_predict = _coerce_positive_int(generation_options.get("num_predict"))
    generation_options["num_predict"] = _resolve_num_predict(
        normalized_mode,
        cleaned_prompt,
        current_num_predict,
    )

    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL_NAME,
        "prompt": cleaned_prompt,
        "stream": bool(stream),
        "options": generation_options,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    if system_prompt and system_prompt.strip():
        payload["system"] = system_prompt.strip()

    return payload


def call_ollama(payload: Mapping[str, Any], timeout: int | None = None) -> dict[str, Any] | str:
    response = _request_ollama(payload, timeout=timeout)

    try:
        raw_text = response.read().decode("utf-8").strip()
    except ValueError:
        raw_text = ""
    finally:
        response.close()

    if not raw_text:
        raise OllamaError("Ollama returned an empty response.")

    try:
        return json.loads(raw_text)
    except ValueError:
        return raw_text


def _call_ollama_stream(
    payload: Mapping[str, Any],
    timeout: int | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    stream_payload = dict(payload)
    stream_payload["stream"] = True

    response = None
    streamed_chunks: list[str] = []
    emitted_any = False

    try:
        response = _request_ollama(stream_payload, timeout=timeout)
    except OllamaError as exc:
        if not emitted_any:
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False
            raw_response = call_ollama(fallback_payload, timeout=timeout)
            return _extract_response_text(raw_response)
        raise OllamaError(f"Could not stream response from Ollama at {OLLAMA_URL}: {exc}") from exc

    try:
        response_status = getattr(response, "status", None)
        if response_status is None:
            response_status = response.getcode()
        if response_status and response_status >= 400:
            detail = response.read().decode("utf-8", errors="replace").strip()
            if detail:
                raise OllamaError(f"Ollama returned HTTP {response_status}: {detail}")
            raise OllamaError(f"Ollama returned HTTP {response_status}.")

        for raw_line in response:
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                chunk_payload = json.loads(line)
            except ValueError as exc:
                if not emitted_any:
                    fallback_payload = dict(payload)
                    fallback_payload["stream"] = False
                    raw_response = call_ollama(fallback_payload, timeout=timeout)
                    return _extract_response_text(raw_response)
                raise OllamaError("Ollama returned malformed streamed JSON.") from exc

            chunk_text = _extract_stream_chunk_text(chunk_payload)
            if chunk_text:
                emitted_any = True
                streamed_chunks.append(chunk_text)
                if on_chunk is not None:
                    on_chunk(chunk_text)

            if isinstance(chunk_payload, Mapping) and chunk_payload.get("done") is True:
                break
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        if not emitted_any:
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False
            raw_response = call_ollama(fallback_payload, timeout=timeout)
            return _extract_response_text(raw_response)
        raise OllamaError(f"Could not stream response from Ollama at {OLLAMA_URL}: {exc}") from exc
    finally:
        if response is not None:
            response.close()

    accumulated_text = "".join(streamed_chunks)
    accumulated_text = accumulated_text.replace("\r\n", "\n").replace("\r", "\n")
    if accumulated_text:
        return accumulated_text

    if not emitted_any:
        fallback_payload = dict(payload)
        fallback_payload["stream"] = False
        raw_response = call_ollama(fallback_payload, timeout=timeout)
        return _extract_response_text(raw_response)

    return ""


def generate_response(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    options: Mapping[str, Any] | None = None,
    timeout: int | None = None,
    stream_callback: Callable[[str], None] | None = None,
    prefer_stream: bool = True,
) -> str:
    normalized_mode = _normalize_mode(mode)
    payload = build_ollama_payload(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        mode=normalized_mode,
        options=options,
        stream=prefer_stream,
    )

    if prefer_stream:
        cleaned_text = _call_ollama_stream(payload, timeout=timeout, on_chunk=stream_callback)
    else:
        raw_response = call_ollama(payload, timeout=timeout)
        cleaned_text = _extract_response_text(raw_response)

    if cleaned_text:
        return cleaned_text

    raise OllamaError("Ollama returned an empty response.")
