from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHAT_HISTORY_DIR = DATA_DIR / "chats"
CHAT_HISTORY_FILE = CHAT_HISTORY_DIR / "chat_history.json"
SAVED_NOTES_FILE = CHAT_HISTORY_DIR / "saved_notes.txt"
GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
_LOADED_ENV_FILES: list[str] = []


def _strip_optional_quotes(value: str) -> str:
    trimmed_value = value.strip()
    if len(trimmed_value) >= 2 and trimmed_value[0] == trimmed_value[-1] and trimmed_value[0] in {"'", '"'}:
        return trimmed_value[1:-1]
    return trimmed_value


def _load_env_file(path: Path, protected_names: set[str]) -> None:
    if not path.exists():
        return

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in raw_lines:
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        line = stripped_line
        if line.startswith("export "):
            line = line[7:].lstrip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_name = key.strip()
        if not env_name:
            continue

        if env_name in protected_names:
            continue

        os.environ[env_name] = _strip_optional_quotes(value)

    _LOADED_ENV_FILES.append(path.name)


def _load_local_environment() -> None:
    protected_names = set(os.environ.keys())
    for env_file_name in (".env", ".env.local"):
        _load_env_file(BASE_DIR / env_file_name, protected_names)


def _read_env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw_value = _read_env(name)
    if not raw_value:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    raw_value = _read_env(name)
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _normalize_choice(value: str, aliases: Mapping[str, str], default: str) -> str:
    normalized_value = value.strip().lower()
    if not normalized_value:
        return default
    return aliases.get(normalized_value, default)


def mask_secret(value: str, visible_prefix: int = 8, visible_suffix: int = 6) -> str:
    cleaned_value = str(value or "")
    if not cleaned_value:
        return ""
    if len(cleaned_value) <= visible_prefix + visible_suffix:
        return cleaned_value
    return f"{cleaned_value[:visible_prefix]}...{cleaned_value[-visible_suffix:]}"


_load_local_environment()

APP_NAME = _read_env("SONIC_APP_NAME", "SONIC") or "SONIC"
APP_DESCRIPTION = _read_env("SONIC_APP_DESCRIPTION", "Offline-first AI workspace") or "Offline-first AI workspace"

DEFAULT_MODE = "assistant"
DEFAULT_MOOD = DEFAULT_MODE

DEFAULT_RUNTIME_PREFERENCE = _normalize_choice(
    _read_env("SONIC_DEFAULT_RUNTIME", "offline"),
    {
        "auto": "auto",
        "local": "offline",
        "offline": "offline",
        "online": "online",
        "live": "online",
        "liveweb": "online",
        "live_web": "online",
        "web": "online",
    },
    "offline",
)

SONIC_HOST = _read_env("SONIC_HOST", "0.0.0.0") or "0.0.0.0"
SONIC_PORT = max(1, min(_read_int_env("SONIC_PORT", 7860), 65535))
SONIC_SHARE = _read_bool_env("SONIC_SHARE", default=False)

OLLAMA_URL = _read_env("OLLAMA_URL", "http://localhost:11434/api/generate") or "http://localhost:11434/api/generate"
OLLAMA_ENDPOINT = OLLAMA_URL
DEFAULT_MODEL_NAME = _read_env("OLLAMA_MODEL", "llama3.2:3b") or "llama3.2:3b"
DEFAULT_MODEL = DEFAULT_MODEL_NAME
OFFLINE_OLLAMA_MODEL = DEFAULT_MODEL_NAME

REQUEST_TIMEOUT_SECONDS = max(5, _read_int_env("REQUEST_TIMEOUT_SECONDS", 120))

# ---------------- NVIDIA IMAGE SETTINGS ----------------
# Stable defaults for SONIC image generation with NVIDIA endpoint
NVIDIA_IMAGE_CONNECT_TIMEOUT_SECONDS = 10
NVIDIA_IMAGE_READ_TIMEOUT_SECONDS = 45
NVIDIA_IMAGE_MAX_RETRIES = 2
NVIDIA_IMAGE_RETRY_BACKOFF_SECONDS = 2
NVIDIA_IMAGE_TOTAL_BUDGET_SECONDS = 60
# ------------------------------------------------------

OLLAMA_KEEP_ALIVE = _read_env("OLLAMA_KEEP_ALIVE", "10m") or "10m"

SERPAPI_API_KEY = _read_env("SERPAPI_API_KEY")
SERPAPI_ENGINE = _read_env("SERPAPI_ENGINE", "google") or "google"
SERPAPI_ENDPOINT = (
    _read_env("SERPAPI_ENDPOINT", "https://serpapi.com/search.json").rstrip("/")
    or "https://serpapi.com/search.json"
)
SERPAPI_HL = _read_env("SERPAPI_HL")
SERPAPI_GL = _read_env("SERPAPI_GL")
SERPAPI_LOCATION = _read_env("SERPAPI_LOCATION")
SERPAPI_REQUEST_TIMEOUT_SECONDS = max(
    5,
    _read_int_env("SERPAPI_REQUEST_TIMEOUT_SECONDS", REQUEST_TIMEOUT_SECONDS),
)
SERPAPI_TRUST_ENV_PROXY = _read_bool_env("SERPAPI_TRUST_ENV_PROXY", default=False)

NVIDIA_API_KEY = (
    _read_env("NVIDIA_API_KEY")
    or _read_env("NVAPI_API_KEY")
    or _read_env("NVIDIA_NIM_API_KEY")
)

NVIDIA_IMAGE_API_ENDPOINT = _read_env(
    "NVIDIA_IMAGE_API_ENDPOINT",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b",
)

NVIDIA_IMAGE_MODEL = _read_env(
    "NVIDIA_IMAGE_MODEL",
    "black-forest-labs/flux.2-klein-4b",
) or "black-forest-labs/flux.2-klein-4b"

_enable_live_web_override = _read_env("ENABLE_LIVE_WEB")
if _enable_live_web_override:
    ENABLE_LIVE_WEB = _read_bool_env("ENABLE_LIVE_WEB", default=False)
else:
    ENABLE_LIVE_WEB = bool(SERPAPI_API_KEY)

ONLINE_PROVIDER_NAME = "serpapi"
LIVE_WEB_PROVIDER_NAME = ONLINE_PROVIDER_NAME
ONLINE_API_KEY = SERPAPI_API_KEY if ENABLE_LIVE_WEB else ""

DEFAULT_GENERATION_OPTIONS = {
    "temperature": 0.35,
    "top_p": 0.9,
    "num_predict": 512,
    "repeat_penalty": 1.08,
}

MODE_GENERATION_OPTIONS = {
    "prompt": {
        "temperature": 0.5,
        "top_p": 0.95,
        "num_predict": 384,
    },
    "summary": {
        "temperature": 0.2,
        "top_p": 0.8,
        "num_predict": 256,
    },
}

RECENT_CONTEXT_LIMIT = 6
HISTORY_PREVIEW_LIMIT = 8
SUMMARY_HISTORY_LIMIT = 10
MAX_DISPLAY_WIDTH = 96

CHAT_HISTORY_PATH = CHAT_HISTORY_FILE
SAVED_NOTES_PATH = SAVED_NOTES_FILE
DEFAULT_OPTIONS = DEFAULT_GENERATION_OPTIONS
LOADED_ENV_FILES = tuple(_LOADED_ENV_FILES)

DEFAULT_LANGUAGE_CODE = _read_env("SONIC_DEFAULT_LANGUAGE", "en").lower() or "en"
SESSION_LANGUAGE_CONTINUITY = _read_bool_env("SONIC_SESSION_LANGUAGE_CONTINUITY", default=True)
VOICE_OUTPUT_DEFAULT = _read_bool_env("SONIC_VOICE_OUTPUT_DEFAULT", default=False)

SUPPORTED_LANGUAGES = {
    "en": {"label": "English", "instruction_label": "English"},
    "hi": {"label": "Hindi", "instruction_label": "Hindi or natural Hinglish"},
    "zh": {"label": "Chinese", "instruction_label": "Chinese"},
    "es": {"label": "Spanish", "instruction_label": "Spanish"},
    "fr": {"label": "French", "instruction_label": "French"},
    "bn": {"label": "Bengali", "instruction_label": "Bengali"},
    "te": {"label": "Telugu", "instruction_label": "Telugu"},
    "mr": {"label": "Marathi", "instruction_label": "Marathi"},
    "ru": {"label": "Russian", "instruction_label": "Russian"},
    "gu": {"label": "Gujarati", "instruction_label": "Gujarati"},
}

if DEFAULT_LANGUAGE_CODE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE_CODE = "en"


def get_generation_options(
    mode: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(DEFAULT_GENERATION_OPTIONS)

    if mode:
        preset = MODE_GENERATION_OPTIONS.get(mode.strip().lower())
        if preset:
            options.update(preset)

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                options[key] = value

    return options