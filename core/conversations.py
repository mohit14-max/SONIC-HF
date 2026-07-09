from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from config import CHAT_HISTORY_DIR, CHAT_HISTORY_FILE
from core.prompts import sanitize_identity_leaks

CONVERSATIONS_FILE = CHAT_HISTORY_DIR / "conversations.json"
UNTITLED_CHAT_TITLE = "New chat"
TITLE_MAX_LENGTH = 52


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _title_from_message(message: str) -> str:
    title = re.sub(r"\s+", " ", _clean_text(message))
    title = title.strip(" \t\n\r\"'")
    if not title:
        return UNTITLED_CHAT_TITLE
    if len(title) <= TITLE_MAX_LENGTH:
        return title
    return title[: TITLE_MAX_LENGTH - 3].rstrip() + "..."


def _default_store() -> dict[str, Any]:
    return {"active_conversation_id": None, "conversations": []}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_store()

    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return _default_store()

    if not raw_text:
        return _default_store()

    try:
        loaded_data = json.loads(raw_text)
    except json.JSONDecodeError:
        return _default_store()

    if isinstance(loaded_data, Mapping):
        conversations = loaded_data.get("conversations")
        if isinstance(conversations, list):
            return {
                "active_conversation_id": loaded_data.get("active_conversation_id"),
                "conversations": conversations,
            }

    if isinstance(loaded_data, list):
        return {"active_conversation_id": None, "conversations": loaded_data}

    return _default_store()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    serialized_data = json.dumps(data, indent=2, ensure_ascii=False)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(serialized_data, encoding="utf-8")
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        path.write_text(serialized_data, encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _sanitize_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, Mapping):
        return None

    role = _clean_text(message.get("role")).lower()
    content = _clean_text(message.get("content"))
    if role not in {"user", "assistant", "system"} or not content:
        return None

    if role == "assistant":
        content = sanitize_identity_leaks(content).strip()

    sanitized: dict[str, Any] = {
        "role": role,
        "content": content,
        "timestamp": _clean_text(message.get("timestamp")) or _timestamp(),
    }
    metadata = message.get("metadata")
    if isinstance(metadata, Mapping):
        sanitized["metadata"] = dict(metadata)

    display_content = _clean_text(message.get("display_content"))
    if display_content:
        sanitized["display_content"] = display_content

    return sanitized


def _sanitize_conversation(conversation: Any) -> dict[str, Any] | None:
    if not isinstance(conversation, Mapping):
        return None

    conversation_id = _clean_text(conversation.get("id")) or _new_id()
    created_at = _clean_text(conversation.get("created_at")) or _timestamp()
    updated_at = _clean_text(conversation.get("updated_at")) or created_at
    raw_messages = conversation.get("messages")
    messages = []
    if isinstance(raw_messages, list):
        messages = [message for item in raw_messages if (message := _sanitize_message(item))]

    title = _clean_text(conversation.get("title"))
    if not title:
        first_user = next((message["content"] for message in messages if message["role"] == "user"), "")
        title = _title_from_message(first_user)

    return {
        "id": conversation_id,
        "title": title or UNTITLED_CHAT_TITLE,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages,
    }


def _legacy_history_entries() -> list[dict[str, Any]]:
    if not CHAT_HISTORY_FILE.exists():
        return []

    try:
        loaded_data = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8").strip() or "[]")
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(loaded_data, list):
        return []

    return [entry for entry in loaded_data if isinstance(entry, Mapping)]


def _conversation_from_legacy_history() -> dict[str, Any] | None:
    legacy_entries = _legacy_history_entries()
    if not legacy_entries:
        return None

    messages: list[dict[str, Any]] = []
    for entry in legacy_entries:
        timestamp = _clean_text(entry.get("timestamp")) or _timestamp()
        user_message = _clean_text(entry.get("user_message") or entry.get("user_input") or entry.get("user"))
        assistant_response = _clean_text(
            entry.get("assistant_response") or entry.get("assistant_reply") or entry.get("assistant")
        )
        if user_message:
            messages.append({"role": "user", "content": user_message, "timestamp": timestamp})
        if assistant_response:
            messages.append(
                {
                    "role": "assistant",
                    "content": sanitize_identity_leaks(assistant_response).strip(),
                    "timestamp": timestamp,
                    "metadata": dict(entry.get("metadata")) if isinstance(entry.get("metadata"), Mapping) else {},
                }
            )

    if not messages:
        return None

    now = _timestamp()
    first_user = next((message["content"] for message in messages if message["role"] == "user"), "")
    return {
        "id": _new_id(),
        "title": _title_from_message(first_user) if first_user else "Imported history",
        "created_at": _clean_text(legacy_entries[0].get("timestamp")) or now,
        "updated_at": _clean_text(legacy_entries[-1].get("timestamp")) or now,
        "messages": messages,
    }


def load_store() -> dict[str, Any]:
    if not CONVERSATIONS_FILE.exists():
        imported_conversation = _conversation_from_legacy_history()
        if imported_conversation is not None:
            store = {
                "active_conversation_id": imported_conversation["id"],
                "conversations": [imported_conversation],
            }
            _write_json(CONVERSATIONS_FILE, store)
            return store

    raw_store = _read_json(CONVERSATIONS_FILE)
    conversations = [
        conversation
        for item in raw_store.get("conversations", [])
        if (conversation := _sanitize_conversation(item))
    ]
    known_ids = {conversation["id"] for conversation in conversations}
    active_id = _clean_text(raw_store.get("active_conversation_id"))
    if active_id not in known_ids:
        active_id = sorted(conversations, key=lambda item: item["updated_at"], reverse=True)[0]["id"] if conversations else None

    return {"active_conversation_id": active_id, "conversations": conversations}


def save_store(store: Mapping[str, Any]) -> None:
    conversations = [
        conversation
        for item in store.get("conversations", [])
        if (conversation := _sanitize_conversation(item))
    ]
    active_id = _clean_text(store.get("active_conversation_id"))
    if active_id and active_id not in {conversation["id"] for conversation in conversations}:
        active_id = None
    _write_json(
        CONVERSATIONS_FILE,
        {
            "active_conversation_id": active_id or None,
            "conversations": conversations,
        },
    )


def list_conversations() -> list[dict[str, Any]]:
    store = load_store()
    return sorted(
        list(store["conversations"]),
        key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""),
        reverse=True,
    )


def get_conversation(conversation_id: str | None) -> dict[str, Any] | None:
    cleaned_id = _clean_text(conversation_id)
    if not cleaned_id:
        return None

    for conversation in load_store()["conversations"]:
        if conversation["id"] == cleaned_id:
            return conversation
    return None


def get_active_conversation() -> dict[str, Any] | None:
    store = load_store()
    active_id = store.get("active_conversation_id")
    return get_conversation(active_id)


def create_conversation(title: str = UNTITLED_CHAT_TITLE) -> dict[str, Any]:
    store = load_store()
    now = _timestamp()
    conversation = {
        "id": _new_id(),
        "title": _clean_text(title) or UNTITLED_CHAT_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    store["conversations"].append(conversation)
    store["active_conversation_id"] = conversation["id"]
    save_store(store)
    return conversation


def ensure_active_conversation(conversation_id: str | None = None) -> dict[str, Any]:
    store = load_store()
    requested_id = _clean_text(conversation_id)
    for conversation in store["conversations"]:
        if requested_id and conversation["id"] == requested_id:
            store["active_conversation_id"] = requested_id
            save_store(store)
            return conversation

    active_id = _clean_text(store.get("active_conversation_id"))
    for conversation in store["conversations"]:
        if active_id and conversation["id"] == active_id:
            return conversation

    if store["conversations"]:
        conversation = list_conversations()[0]
        store["active_conversation_id"] = conversation["id"]
        save_store(store)
        return conversation

    return create_conversation()


def set_active_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return ensure_active_conversation()

    store = load_store()
    store["active_conversation_id"] = conversation["id"]
    save_store(store)
    return conversation


def append_exchange(
    conversation_id: str | None,
    user_message: str,
    assistant_response: str,
    metadata: Mapping[str, Any] | None = None,
    display_response: str | None = None,
) -> dict[str, Any]:
    conversation = ensure_active_conversation(conversation_id)
    store = load_store()
    now = _timestamp()
    for item in store["conversations"]:
        if item["id"] != conversation["id"]:
            continue

        item.setdefault("messages", [])
        item["messages"].append({"role": "user", "content": _clean_text(user_message), "timestamp": now})

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": sanitize_identity_leaks(_clean_text(assistant_response)).strip(),
            "timestamp": now,
        }
        if metadata:
            assistant_message["metadata"] = dict(metadata)
        if display_response:
            assistant_message["display_content"] = display_response
        item["messages"].append(assistant_message)

        if not item.get("title") or item.get("title") == UNTITLED_CHAT_TITLE:
            item["title"] = _title_from_message(user_message)
        item["updated_at"] = now
        store["active_conversation_id"] = item["id"]
        save_store(store)
        return _sanitize_conversation(item) or item

    return conversation


def clear_conversation(conversation_id: str | None) -> dict[str, Any]:
    conversation = ensure_active_conversation(conversation_id)
    store = load_store()
    now = _timestamp()
    for item in store["conversations"]:
        if item["id"] == conversation["id"]:
            item["messages"] = []
            item["title"] = UNTITLED_CHAT_TITLE
            item["updated_at"] = now
            store["active_conversation_id"] = item["id"]
            save_store(store)
            return _sanitize_conversation(item) or item
    return conversation


def delete_conversation(conversation_id: str | None) -> dict[str, Any] | None:
    cleaned_id = _clean_text(conversation_id)
    if not cleaned_id:
        return ensure_active_conversation()

    store = load_store()
    remaining = [conversation for conversation in store["conversations"] if conversation["id"] != cleaned_id]
    store["conversations"] = remaining

    if remaining:
        next_conversation = sorted(
            remaining,
            key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""),
            reverse=True,
        )[0]
        store["active_conversation_id"] = next_conversation["id"]
        save_store(store)
        return _sanitize_conversation(next_conversation) or next_conversation

    store["active_conversation_id"] = None
    save_store(store)
    return create_conversation()


def conversation_messages_to_backend_history(messages: list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not messages:
        return []

    history: list[dict[str, Any]] = []
    pending_user: Mapping[str, Any] | None = None
    for message in messages:
        role = _clean_text(message.get("role")).lower()
        content = _clean_text(message.get("content"))
        if not content:
            continue

        if role == "user":
            pending_user = message
            continue

        if role == "assistant":
            entry: dict[str, Any] = {
                "timestamp": _clean_text(message.get("timestamp")) or _timestamp(),
                "user_message": _clean_text(pending_user.get("content")) if pending_user else "",
                "assistant_response": content,
            }
            metadata = message.get("metadata")
            if isinstance(metadata, Mapping):
                entry["metadata"] = dict(metadata)
            history.append(entry)
            pending_user = None

    return history
