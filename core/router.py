from __future__ import annotations

import re
from dataclasses import dataclass

# Mode import removed


@dataclass(frozen=True, slots=True)
class RouteResult:
    action: str
    mode: str
    raw_input: str
    command: str = ""
    content: str = ""
    task_name: str = ""
    message: str = ""


TASK_COMMANDS = {
    "/prompt": "prompt",
}

UTILITY_COMMANDS = {
    "/history": "history",
    "/clear": "clear",
    "/save-note": "save_note",
    "/summarize": "summarize",
}

QUERY_TYPE_WEB_LOOKUP = "web_lookup"
QUERY_TYPE_GENERATIVE_TASK = "generative_task"
QUERY_TYPE_HYBRID_TASK = "hybrid_task"

_LIVE_LOOKUP_HINTS = (
    "today",
    "current weather",
    "weather",
    "forecast",
    "temperature",
    "humidity",
    "wind",
    "current time",
    "time now",
    "today date",
    "current date",
    "date today",
    "now",
    "right now",
    "latest",
    "recent",
    "breaking",
    "news",
    "headline",
    "score",
    "scores",
    "result",
    "results",
    "winner",
    "won",
    "match",
    "game",
    "ipl",
    "fifa",
    "world cup",
    "bitcoin",
    "crypto",
    "price",
    "prices",
    "stock",
    "stocks",
    "share price",
    "exchange rate",
    "president",
    "prime minister",
    "ceo",
    "launch date",
    "release date",
    "aaj",
    "aaj ki date",
    "date kya",
    "kaun jeeta",
    "kisne jeeta",
    "winner kaun",
    "आज",
    "तारीख",
    "कौन जीता",
    "आज की तारीख",
    "আজ",
    "তারিখ",
    "ఈరోజు",
    "తేదీ",
    "आजची तारीख",
    "આજ",
    "તારીખ",
    "сегодня",
    "текущ",
    "hoy",
    "actual",
    "aujourd",
    "actuel",
)

_LIVE_LOOKUP_PATTERNS = (
    r"\bwho won\b",
    r"\bwho is the current\b",
    r"\bcurrent (?:president|prime minister|ceo|weather|date|time)\b",
    r"\blatest\b",
    r"\btoday\b",
    r"\bright now\b",
)

_GENERATION_HINTS = (
    "write",
    "create",
    "build",
    "make",
    "draft",
    "generate",
    "explain",
    "summarize",
    "summary",
    "code",
    "program",
    "app",
    "website",
    "html",
    "css",
    "javascript",
    "python",
    "sql",
    "email",
    "essay",
    "plan",
    "outline",
    "rewrite",
    "fix",
    "debug",
    "solve",
    "compare",
    "analyze",
    "teach me",
    "help me",
    "walk me through",
    "how do i",
    "how to",
    "what should i",
    "should i",
    "what do you recommend",
    "what would you recommend",
)

_HYBRID_HINTS = (
    "should i",
    "what should i",
    "what do you recommend",
    "what would you recommend",
    "what if",
    "and also",
    "also tell me",
    "then write",
    "write it",
    "explain",
    "install",
    "wear",
    "worry",
)


def _normalize_query_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _count_phrase_matches(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def _count_pattern_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def classify_query(message: str) -> str:
    cleaned_message = _normalize_query_text(message)
    if not cleaned_message:
        return QUERY_TYPE_GENERATIVE_TASK

    live_score = _count_phrase_matches(cleaned_message, _LIVE_LOOKUP_HINTS)
    live_score += _count_pattern_matches(cleaned_message, _LIVE_LOOKUP_PATTERNS)

    generative_score = _count_phrase_matches(cleaned_message, _GENERATION_HINTS)
    hybrid_score = _count_phrase_matches(cleaned_message, _HYBRID_HINTS)

    if live_score > 0 and (generative_score > 0 or hybrid_score > 0):
        return QUERY_TYPE_HYBRID_TASK

    if live_score > 0:
        return QUERY_TYPE_WEB_LOOKUP

    return QUERY_TYPE_GENERATIVE_TASK


def should_use_web_search(message: str) -> bool:
    return classify_query(message) in {QUERY_TYPE_WEB_LOOKUP, QUERY_TYPE_HYBRID_TASK}


def _split_command(user_input: str) -> tuple[str, str]:
    stripped_input = user_input.strip()
    if not stripped_input:
        return "", ""

    parts = stripped_input.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    return command, argument


def route_user_input(user_input: str, current_mode: str) -> RouteResult:
    stripped_input = user_input.strip()
    if not stripped_input:
        return RouteResult(action="empty", mode=current_mode, raw_input=user_input)

    lowered_input = stripped_input.lower()
    if lowered_input in {"help", "/help"}:
        return RouteResult(action="help", mode=current_mode, raw_input=user_input)

    if lowered_input in {"exit", "quit", "/exit", "/quit"}:
        return RouteResult(action="exit", mode=current_mode, raw_input=user_input)

    command, argument = _split_command(stripped_input)



    if command in TASK_COMMANDS:
        if not argument:
            return RouteResult(
                action="error",
                mode=current_mode,
                raw_input=user_input,
                command=command,
                message=f"Usage: {command} <text>",
            )

        return RouteResult(
            action="task",
            mode=current_mode,
            raw_input=user_input,
            command=command,
            content=argument,
            task_name=TASK_COMMANDS[command],
        )

    if command == "/image":
        if not argument:
            return RouteResult(
                action="error",
                mode=current_mode,
                raw_input=user_input,
                command=command,
                message="Usage: /image <text prompt>",
            )

        return RouteResult(
            action="image",
            mode=current_mode,
            raw_input=user_input,
            command=command,
            content=argument,
            task_name="image",
        )

    if command in UTILITY_COMMANDS:
        return RouteResult(
            action=UTILITY_COMMANDS[command],
            mode=current_mode,
            raw_input=user_input,
            command=command,
            content=argument,
        )

    if stripped_input.startswith("/"):
        return RouteResult(
            action="error",
            mode=current_mode,
            raw_input=user_input,
            command=command or stripped_input,
            message="Unknown command. Type help to see available commands.",
        )

    return RouteResult(
        action="chat",
        mode=current_mode,
        raw_input=user_input,
        content=stripped_input,
    )
