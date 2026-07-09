from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from config import DEFAULT_LANGUAGE_CODE, SUPPORTED_LANGUAGES


@dataclass(frozen=True, slots=True)
class LanguageContext:
    code: str
    label: str
    instruction_label: str
    confidence: float
    source: str
    is_ambiguous: bool = False


SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "hi": ((0x0900, 0x097F),),
    "bn": ((0x0980, 0x09FF),),
    "te": ((0x0C00, 0x0C7F),),
    "gu": ((0x0A80, 0x0AFF),),
    "ru": ((0x0400, 0x04FF),),
    "zh": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
}

LANGUAGE_NAME_ALIASES: dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "hinglish": "hi",
    "chinese": "zh",
    "mandarin": "zh",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "bengali": "bn",
    "bangla": "bn",
    "telugu": "te",
    "marathi": "mr",
    "russian": "ru",
    "gujarati": "gu",
}

EXPLICIT_LANGUAGE_PATTERNS = (
    r"\b(?:answer|reply|respond|write|explain|say|translate(?: this)?)\s+(?:this\s+|it\s+)?(?:in|to)\s+([a-zA-ZÀ-ÿ]+)\b",
    r"\b(?:in|to)\s+([a-zA-ZÀ-ÿ]+)\s+(?:please|pls|language)\b",
)

ROMAN_HINDI_HINTS = {
    "aaj",
    "abhi",
    "achha",
    "acha",
    "aur",
    "bata",
    "batao",
    "bhai",
    "bolo",
    "hai",
    "haan",
    "han",
    "ka",
    "kya",
    "kaun",
    "ki",
    "ko",
    "likh",
    "likho",
    "me",
    "mein",
    "nahi",
    "nhi",
    "tha",
    "thi",
    "tum",
    "winner",
}

SPANISH_HINTS = {
    "que",
    "qué",
    "como",
    "cómo",
    "hola",
    "gracias",
    "por",
    "favor",
    "escribe",
    "explica",
    "dime",
    "quien",
    "quién",
    "fecha",
}

FRENCH_HINTS = {
    "bonjour",
    "merci",
    "avec",
    "pour",
    "quoi",
    "qui",
    "écris",
    "ecris",
    "explique",
    "réponds",
    "reponds",
    "aujourd",
    "date",
}

MARATHI_HINTS = {
    "आहे",
    "आहेत",
    "काय",
    "कोण",
    "मला",
    "मध्ये",
    "सांगा",
}

HINDI_HINTS = {
    "क्या",
    "कौन",
    "है",
    "हैं",
    "मुझे",
    "बताओ",
    "लिख",
    "लिखो",
}

AMBIGUOUS_SHORT_MESSAGES = {
    "ok",
    "okay",
    "hmm",
    "hm",
    "haan",
    "han",
    "yes",
    "no",
    "aur",
    "aur?",
    "more",
    "continue",
}


def _language_entry(code: str) -> Mapping[str, str]:
    return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE_CODE])


def _context(
    code: str,
    *,
    confidence: float,
    source: str,
    is_ambiguous: bool = False,
) -> LanguageContext:
    normalized_code = code if code in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE_CODE
    entry = _language_entry(normalized_code)
    return LanguageContext(
        code=normalized_code,
        label=str(entry["label"]),
        instruction_label=str(entry.get("instruction_label") or entry["label"]),
        confidence=confidence,
        source=source,
        is_ambiguous=is_ambiguous,
    )


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())


def _script_counts(text: str) -> dict[str, int]:
    counts = {code: 0 for code in SCRIPT_RANGES}
    for char in text or "":
        point = ord(char)
        for code, ranges in SCRIPT_RANGES.items():
            if any(start <= point <= end for start, end in ranges):
                counts[code] += 1
                break
    return counts


def _detect_explicit_language(text: str) -> str:
    lowered_text = _clean_text(text).lower()
    for pattern in EXPLICIT_LANGUAGE_PATTERNS:
        for match in re.finditer(pattern, lowered_text, flags=re.IGNORECASE):
            candidate = match.group(1).strip().lower()
            code = LANGUAGE_NAME_ALIASES.get(candidate)
            if code:
                return code

    for name, code in LANGUAGE_NAME_ALIASES.items():
        if re.search(rf"\b{name}\s+(?:language|me|mein)\b", lowered_text, flags=re.IGNORECASE):
            return code

    return ""


def detect_message_language(message: str) -> LanguageContext:
    cleaned_message = _clean_text(message)
    if not cleaned_message:
        return _context(DEFAULT_LANGUAGE_CODE, confidence=0.0, source="empty", is_ambiguous=True)

    explicit_code = _detect_explicit_language(cleaned_message)
    if explicit_code:
        return _context(explicit_code, confidence=1.0, source="explicit_instruction")

    counts = _script_counts(cleaned_message)
    script_code, script_count = max(counts.items(), key=lambda item: item[1])
    if script_count > 0:
        if script_code == "hi":
            marathi_hits = sum(1 for hint in MARATHI_HINTS if hint in cleaned_message)
            hindi_hits = sum(1 for hint in HINDI_HINTS if hint in cleaned_message)
            if marathi_hits > hindi_hits:
                return _context("mr", confidence=0.92, source="devanagari_marathi")
            return _context("hi", confidence=0.9, source="devanagari_hindi")
        return _context(script_code, confidence=0.95, source="script")

    tokens = _word_tokens(cleaned_message)
    token_count = len(tokens)
    normalized_short = cleaned_message.lower().strip(" .,!?:;")
    if token_count <= 2 and normalized_short in AMBIGUOUS_SHORT_MESSAGES:
        return _context(DEFAULT_LANGUAGE_CODE, confidence=0.2, source="short_ambiguous", is_ambiguous=True)

    if not tokens:
        return _context(DEFAULT_LANGUAGE_CODE, confidence=0.15, source="no_language_tokens", is_ambiguous=True)

    roman_hindi_hits = sum(1 for token in tokens if token in ROMAN_HINDI_HINTS)
    spanish_hits = sum(1 for token in tokens if token in SPANISH_HINTS)
    french_hits = sum(1 for token in tokens if token in FRENCH_HINTS)

    if roman_hindi_hits >= 2 or (
        roman_hindi_hits >= 1 and any(token in {"bhai", "kya", "kaun", "likh", "bata"} for token in tokens)
    ):
        return _context("hi", confidence=0.78, source="roman_hindi")

    if spanish_hits >= 2 or any(char in cleaned_message.lower() for char in "¿¡ñ"):
        return _context("es", confidence=0.76, source="latin_spanish")

    if french_hits >= 2 or any(char in cleaned_message.lower() for char in "àâçèéêëîïôûùüÿœ"):
        return _context("fr", confidence=0.76, source="latin_french")

    return _context("en", confidence=0.65, source="latin_default")


def _metadata_language(entry: Mapping[str, Any]) -> str:
    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping):
        code = str(metadata.get("language_code") or metadata.get("detected_language") or "").strip().lower()
        if code in SUPPORTED_LANGUAGES:
            return code

    code = str(entry.get("language_code") or entry.get("detected_language") or "").strip().lower()
    if code in SUPPORTED_LANGUAGES:
        return code

    return ""


def previous_language_from_history(history: Sequence[Any] | None) -> LanguageContext | None:
    if not history:
        return None

    for entry in reversed(list(history)):
        if isinstance(entry, Mapping):
            metadata_code = _metadata_language(entry)
            if metadata_code:
                return _context(metadata_code, confidence=0.8, source="history_metadata")

            user_message = str(
                entry.get("user_message")
                or entry.get("user_input")
                or entry.get("user")
                or (entry.get("content") if str(entry.get("role") or "").lower() == "user" else "")
                or ""
            )
            detected = detect_message_language(user_message)
            if not detected.is_ambiguous and detected.confidence >= 0.6:
                return _context(detected.code, confidence=0.65, source="history_detection")
            continue

        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)) and entry:
            detected = detect_message_language(str(entry[0]))
            if not detected.is_ambiguous and detected.confidence >= 0.6:
                return _context(detected.code, confidence=0.65, source="history_detection")

    return None


def resolve_language_context(
    message: str,
    history: Sequence[Any] | None = None,
) -> LanguageContext:
    current_context = detect_message_language(message)
    previous_context = previous_language_from_history(history)

    if current_context.is_ambiguous and previous_context is not None:
        return _context(previous_context.code, confidence=previous_context.confidence, source="session_continuity")

    if current_context.confidence < 0.5 and previous_context is not None:
        return _context(previous_context.code, confidence=previous_context.confidence, source="session_continuity")

    return current_context


def build_language_instruction(language_context: LanguageContext | None) -> str:
    context = language_context or _context(DEFAULT_LANGUAGE_CODE, confidence=0.0, source="fallback")
    return (
        f"Conversation language: {context.instruction_label} ({context.code}). "
        "Reply in this same language and tone by default. "
        "Preserve mixed-language style naturally, such as Hinglish, when that is how the user writes. "
        "If the user explicitly asks for a different output language, follow that instruction. "
        "For coding requests, keep code syntax, identifiers, APIs, and programming keywords unchanged; "
        "localize only the surrounding explanation when explanation is needed."
    )


def language_metadata(language_context: LanguageContext | None) -> dict[str, Any]:
    context = language_context or _context(DEFAULT_LANGUAGE_CODE, confidence=0.0, source="fallback")
    return {
        "language_code": context.code,
        "language_label": context.label,
        "language_source": context.source,
        "language_confidence": context.confidence,
        "language_ambiguous": context.is_ambiguous,
    }
