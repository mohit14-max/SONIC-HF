from __future__ import annotations

import base64
import re
from dataclasses import dataclass

from config import DEFAULT_LANGUAGE_CODE, SUPPORTED_LANGUAGES


TTS_LANGUAGE_LOCALES: dict[str, tuple[str, ...]] = {
    "en": ("en-US", "en-GB", "en-IN"),
    "hi": ("hi-IN",),
    "zh": ("zh-CN", "zh-TW", "zh-HK"),
    "es": ("es-ES", "es-MX", "es-US"),
    "fr": ("fr-FR", "fr-CA"),
    "bn": ("bn-IN", "bn-BD"),
    "te": ("te-IN",),
    "mr": ("mr-IN",),
    "ru": ("ru-RU",),
    "gu": ("gu-IN",),
}

CODE_NOTICE_BY_LANGUAGE: dict[str, str] = {
    "en": "I've included the full code in the chat.",
    "hi": "Maine poora code chat mein diya hai.",
    "zh": "完整代码已放在聊天中。",
    "es": "He incluido el código completo en el chat.",
    "fr": "J'ai inclus le code complet dans le chat.",
    "bn": "আমি চ্যাটে সম্পূর্ণ কোডটি দিয়েছি।",
    "te": "పూర్తి కోడ్‌ను చాట్‌లో ఇచ్చాను.",
    "mr": "मी पूर्ण कोड चॅटमध्ये दिला आहे.",
    "ru": "Полный код приведён в чате.",
    "gu": "મેં સંપૂર્ણ કોડ ચેટમાં આપ્યો છે.",
}

_FENCED_CODE_PATTERN = re.compile(r"```[\s\S]*?```", flags=re.MULTILINE)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_URL_PATTERN = re.compile(r"https?://\S+")
_SOURCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:sources?|references?)\s*:?\s*$"
)


@dataclass(frozen=True, slots=True)
class SpeechPayload:
    text: str
    language_code: str
    locale_preferences: tuple[str, ...]
    skipped_code: bool = False


def normalize_tts_language(language_code: str | None) -> str:
    normalized = str(language_code or "").strip().lower().replace("_", "-")
    base_code = normalized.split("-", 1)[0]
    if base_code in SUPPORTED_LANGUAGES:
        return base_code
    return DEFAULT_LANGUAGE_CODE


def _strip_sources_section(text: str) -> str:
    match = _SOURCE_HEADING_PATTERN.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def prepare_spoken_text(reply_text: str, language_code: str | None = None) -> tuple[str, bool]:
    language = normalize_tts_language(language_code)
    text = str(reply_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", False

    skipped_code = bool(_FENCED_CODE_PATTERN.search(text))
    text = _FENCED_CODE_PATTERN.sub("\n", text)
    text = _strip_sources_section(text)
    text = _MARKDOWN_IMAGE_PATTERN.sub("", text)
    text = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = _URL_PATTERN.sub("", text)
    text = re.sub(r"(?m)^\s{4,}\S.*$", "", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|>|[-*+]|\d+[.)])\s*", "", text)
    text = re.sub(r"[*_~]+", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    if skipped_code:
        code_notice = CODE_NOTICE_BY_LANGUAGE.get(language, CODE_NOTICE_BY_LANGUAGE["en"])
        text = f"{text}. {code_notice}" if text else code_notice

    return text.strip(), skipped_code


def build_speech_payload(reply_text: str, language_code: str | None = None) -> SpeechPayload:
    language = normalize_tts_language(language_code)
    spoken_text, skipped_code = prepare_spoken_text(reply_text, language)
    return SpeechPayload(
        text=spoken_text,
        language_code=language,
        locale_preferences=TTS_LANGUAGE_LOCALES.get(
            language,
            TTS_LANGUAGE_LOCALES[DEFAULT_LANGUAGE_CODE],
        ),
        skipped_code=skipped_code,
    )


def encode_speech_text(text: str) -> str:
    return base64.b64encode(str(text or "").encode("utf-8")).decode("ascii")
