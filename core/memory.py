from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from config import CHAT_HISTORY_DIR, CHAT_HISTORY_FILE, SAVED_NOTES_FILE
from core.prompts import sanitize_identity_leaks


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return []

    if not raw_text:
        return []

    try:
        loaded_data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    if isinstance(loaded_data, list):
        return [entry for entry in loaded_data if isinstance(entry, dict)]

    return []


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sanitized_entry = dict(entry)
    if "assistant_response" in sanitized_entry:
        sanitized_entry["assistant_response"] = sanitize_identity_leaks(
            str(sanitized_entry.get("assistant_response") or "")
        ).strip()
    return sanitized_entry


def ensure_chat_history_file() -> Path:
    _ensure_parent(CHAT_HISTORY_FILE)
    if not CHAT_HISTORY_FILE.exists():
        CHAT_HISTORY_FILE.write_text("[]", encoding="utf-8")
    return CHAT_HISTORY_FILE


def load_history(limit: int | None = None) -> list[dict[str, Any]]:
    ensure_chat_history_file()
    history = [_sanitize_history_entry(entry) for entry in _safe_read_json(CHAT_HISTORY_FILE)]

    if limit is None or limit <= 0:
        return history

    return history[-limit:]


def get_recent_context(limit: int = 6) -> list[dict[str, Any]]:
    return load_history(limit=limit)


def save_message(
    mode: str,
    user_message: str,
    assistant_response: str,
    route_type: str = "chat",
    task_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_chat_history_file()
    history = load_history()

    record: dict[str, Any] = {
        "timestamp": _timestamp(),
        "mode": mode,
        "user_message": user_message,
        "assistant_response": sanitize_identity_leaks(assistant_response).strip(),
        "route_type": route_type,
    }
    if task_name:
        record["task_name"] = task_name
    if metadata:
        record["metadata"] = dict(metadata)

    history.append(record)
    _write_json(CHAT_HISTORY_FILE, history)
    return record


def save_chat_record(
    user_input: str,
    assistant_reply: str,
    selected_mood: str,
    route_type: str | None = None,
    task_name: str | None = None,
    effective_mode: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return save_message(
        mode=effective_mode or selected_mood,
        user_message=user_input,
        assistant_response=assistant_reply,
        route_type=route_type or "chat",
        task_name=task_name,
        metadata=metadata,
    )


def clear_history() -> int:
    ensure_chat_history_file()
    previous_history = load_history()
    _write_json(CHAT_HISTORY_FILE, [])
    return len(previous_history)


def ensure_saved_notes_file() -> Path:
    _ensure_parent(SAVED_NOTES_FILE)
    if not SAVED_NOTES_FILE.exists():
        SAVED_NOTES_FILE.write_text("", encoding="utf-8")
    return SAVED_NOTES_FILE


def save_note(note_text: str) -> Path:
    cleaned_text = (note_text or "").strip()
    if not cleaned_text:
        raise ValueError("Note text cannot be empty.")

    ensure_saved_notes_file()
    note_block = f"[{_timestamp()}] {cleaned_text}\n\n"
    with SAVED_NOTES_FILE.open("a", encoding="utf-8") as note_file:
        note_file.write(note_block)
    return SAVED_NOTES_FILE
