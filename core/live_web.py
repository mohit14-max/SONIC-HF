from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from core.language import LanguageContext, resolve_language_context
from core.router import (
    QUERY_TYPE_GENERATIVE_TASK,
    QUERY_TYPE_HYBRID_TASK,
    QUERY_TYPE_WEB_LOOKUP,
    classify_query,
)

from config import (
    ENABLE_LIVE_WEB,
    SERPAPI_API_KEY,
    SERPAPI_ENDPOINT,
    SERPAPI_ENGINE,
    SERPAPI_GL,
    SERPAPI_HL,
    SERPAPI_LOCATION,
    SERPAPI_REQUEST_TIMEOUT_SECONDS,
    SERPAPI_TRUST_ENV_PROXY,
    mask_secret,
)

logger = logging.getLogger(__name__)

RESULT_LIMIT = 6
DETAIL_LIMIT = 4
SOURCE_LIMIT = 3
FALLBACK_POINT_LIMIT = 3
WEB_LABELS = {
    "en": {
        "sources": "Sources:",
        "quick_details": "Quick details:",
        "no_clear_answer": "I couldn't find a clear current answer for that query.",
        "limited_detail": "I found current information, but there was little structured detail to add.",
        "hybrid_followup": "If you want, I can also help interpret this or turn it into a next step.",
        "fresh_not_found": "I couldn't find fresh results for that query.",
    },
    "hi": {
        "sources": "Sources:",
        "quick_details": "Quick details:",
        "no_clear_answer": "Is query ke liye mujhe clear current answer nahi mila.",
        "limited_detail": "Current info mila, par add karne layak structured detail kam thi.",
        "hybrid_followup": "Chaaho to main isko interpret karke next step bhi bana sakta hoon.",
        "fresh_not_found": "Is query ke liye fresh results nahi mile.",
    },
    "zh": {
        "sources": "Sources:",
        "quick_details": "要点:",
        "no_clear_answer": "我没有找到这个问题的明确最新答案。",
        "limited_detail": "我找到了最新信息，但可补充的结构化细节不多。",
        "hybrid_followup": "如果需要，我也可以帮你解读这些信息或整理下一步。",
        "fresh_not_found": "我没有找到这个查询的最新结果。",
    },
    "es": {
        "sources": "Sources:",
        "quick_details": "Detalles rápidos:",
        "no_clear_answer": "No encontré una respuesta actual clara para esa consulta.",
        "limited_detail": "Encontré información actual, pero había pocos detalles estructurados para añadir.",
        "hybrid_followup": "También puedo ayudarte a interpretarlo o convertirlo en un siguiente paso.",
        "fresh_not_found": "No encontré resultados recientes para esa consulta.",
    },
    "fr": {
        "sources": "Sources:",
        "quick_details": "Détails rapides:",
        "no_clear_answer": "Je n'ai pas trouvé de réponse actuelle claire pour cette requête.",
        "limited_detail": "J'ai trouvé des informations actuelles, mais peu de détails structurés à ajouter.",
        "hybrid_followup": "Je peux aussi t'aider à l'interpréter ou à en faire une prochaine étape.",
        "fresh_not_found": "Je n'ai pas trouvé de résultats récents pour cette requête.",
    },
    "bn": {
        "sources": "Sources:",
        "quick_details": "দ্রুত তথ্য:",
        "no_clear_answer": "এই প্রশ্নের স্পষ্ট সাম্প্রতিক উত্তর খুঁজে পাইনি।",
        "limited_detail": "সাম্প্রতিক তথ্য পেয়েছি, কিন্তু যোগ করার মতো কাঠামোবদ্ধ বিস্তারিত কম ছিল।",
        "hybrid_followup": "চাইলে আমি এগুলো ব্যাখ্যা করে পরের ধাপও সাজিয়ে দিতে পারি।",
        "fresh_not_found": "এই প্রশ্নের নতুন ফলাফল খুঁজে পাইনি।",
    },
    "te": {
        "sources": "Sources:",
        "quick_details": "త్వరిత వివరాలు:",
        "no_clear_answer": "ఈ ప్రశ్నకు స్పష్టమైన తాజా సమాధానం దొరకలేదు.",
        "limited_detail": "తాజా సమాచారం దొరికింది, కానీ జోడించడానికి నిర్మిత వివరాలు తక్కువగా ఉన్నాయి.",
        "hybrid_followup": "కావాలంటే దీన్ని అర్థం చేసుకుని తదుపరి చర్యగా కూడా మార్చగలను.",
        "fresh_not_found": "ఈ ప్రశ్నకు తాజా ఫలితాలు దొరకలేదు.",
    },
    "mr": {
        "sources": "Sources:",
        "quick_details": "झटपट तपशील:",
        "no_clear_answer": "या प्रश्नासाठी स्पष्ट अद्ययावत उत्तर सापडले नाही.",
        "limited_detail": "अद्ययावत माहिती मिळाली, पण जोडण्यासारखे रचलेले तपशील कमी होते.",
        "hybrid_followup": "हवे असल्यास मी याचा अर्थ लावून पुढची पायरीही सांगू शकतो.",
        "fresh_not_found": "या प्रश्नासाठी ताजे निकाल सापडले नाहीत.",
    },
    "ru": {
        "sources": "Sources:",
        "quick_details": "Краткие детали:",
        "no_clear_answer": "Я не нашел четкого актуального ответа на этот запрос.",
        "limited_detail": "Актуальная информация нашлась, но структурированных деталей было немного.",
        "hybrid_followup": "Я также могу помочь это интерпретировать или превратить в следующий шаг.",
        "fresh_not_found": "Я не нашел свежих результатов по этому запросу.",
    },
    "gu": {
        "sources": "Sources:",
        "quick_details": "ઝડપી વિગતો:",
        "no_clear_answer": "આ પ્રશ્ન માટે સ્પષ્ટ તાજો જવાબ મળ્યો નથી.",
        "limited_detail": "તાજી માહિતી મળી, પણ ઉમેરવા જેવી રચાયેલ વિગતો ઓછી હતી.",
        "hybrid_followup": "જો જોઈએ તો હું તેને સમજાવીને આગળનું પગલું પણ બનાવી શકું.",
        "fresh_not_found": "આ પ્રશ્ન માટે તાજા પરિણામો મળ્યા નથી.",
    },
}
FOLLOW_UP_HINTS = (
    "and ",
    "also ",
    "what about",
    "how about",
    "what's its",
    "what is its",
    "who is its",
    "where is it",
    "when did it",
    "when was it",
    "is it",
    "does it",
    "did it",
    "they ",
    "them ",
    "that ",
    "this ",
    "it ",
)
COMPARISON_HINTS = (
    "best ",
    "compare",
    "comparison",
    " vs ",
    " versus ",
    "difference",
    "top ",
    "under ",
    "review",
    "research",
)
PROMPT_LEAK_PATTERNS = (
    r"<unk>",
    r"^\s*okay[, ]+the user\b",
    r"\bthe user just asked\b",
    r"\bthe user asked\b",
    r"\buser just asked\b",
    r"\bi(?:'m| am)? going to\b",
    r"\blet me think\b",
    r"\bhidden prompt\b",
    r"\bscratchpad\b",
    r"\bchain[- ]of[- ]thought\b",
    r"\binternal reasoning\b",
    r"\binternal prompt\b",
    r"\bdraft answer\b",
    r"\braw normalization\b",
    r"\bintermediate summary\b",
    r"\btoken junk\b",
    r"^\s*(system|assistant|user|analysis|reasoning|thoughts?)\s*:",
)
ROLE_LINE_PATTERN = re.compile(r"^\s*(system|assistant|user|analysis|reasoning|thoughts?)\s*:", re.IGNORECASE)
ANGLE_TOKEN_PATTERN = re.compile(r"</?unk>|</?s>|<\|.*?\|>", re.IGNORECASE)
PROMPT_LEAK_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in PROMPT_LEAK_PATTERNS)


class LiveWebConfigError(RuntimeError):
    pass


class LiveWebAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.detail = detail


def is_live_web_configured() -> bool:
    return bool(
        ENABLE_LIVE_WEB
        and SERPAPI_API_KEY.strip()
        and SERPAPI_ENDPOINT.strip()
        and SERPAPI_ENGINE.strip()
    )


def _build_http_session() -> Any:
    return (
        urllib_request.build_opener()
        if SERPAPI_TRUST_ENV_PROXY
        else urllib_request.build_opener(urllib_request.ProxyHandler({}))
    )


def _clean_text(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").replace("\r", "\n").split())


def _clean_multiline_text(text: str) -> str:
    normalized_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized_text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _strip_leak_tokens(text: str) -> str:
    return ANGLE_TOKEN_PATTERN.sub(" ", text or "")


def _looks_like_prompt_leak(text: str) -> bool:
    cleaned_text = _clean_multiline_text(_strip_leak_tokens(text))
    if not cleaned_text:
        return False

    if any(regex.search(cleaned_text) for regex in PROMPT_LEAK_REGEXES):
        return True

    lowered_text = cleaned_text.lower()
    if lowered_text.count("<") >= 3 and lowered_text.count(">") >= 3:
        return True

    return False


def _sanitize_live_web_fragment(text: str) -> str:
    cleaned_text = _clean_multiline_text(_strip_leak_tokens(text))
    if not cleaned_text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in cleaned_text.splitlines():
        line = _clean_text(raw_line)
        if not line:
            continue
        if _looks_like_prompt_leak(line):
            continue
        if ROLE_LINE_PATTERN.match(line):
            continue
        cleaned_lines.append(line)

    sanitized_text = "\n".join(cleaned_lines).strip()
    if not sanitized_text or _looks_like_prompt_leak(sanitized_text):
        return ""

    return sanitized_text


def sanitize_live_web_output(text: str, *, query: str = "") -> str:
    del query

    cleaned_text = _sanitize_live_web_fragment(text)
    if not cleaned_text:
        return ""

    blocks = [block.strip() for block in cleaned_text.split("\n\n") if block.strip()]
    sanitized_blocks: list[str] = []

    for block in blocks:
        if block.lower() == "sources:":
            continue
        if _looks_like_prompt_leak(block):
            continue
        sanitized_blocks.append(block)

    sanitized_text = "\n\n".join(sanitized_blocks).strip()
    if not sanitized_text or _looks_like_prompt_leak(sanitized_text):
        return ""

    return sanitized_text


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _to_text(value)
        if text:
            return text
    return ""


def _to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return _clean_text(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, Mapping):
        for key in ("text", "snippet", "title", "name", "value", "answer", "description"):
            text = _to_text(value.get(key))
            if text:
                return text
        return ""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_to_text(item) for item in value]
        return _clean_text(" ".join(part for part in parts if part))

    return _clean_text(str(value))


def _domain_from_url(url: str) -> str:
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        return ""

    try:
        parsed_url = urlparse(cleaned_url)
    except ValueError:
        return ""

    domain = parsed_url.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _friendly_label(key: str) -> str:
    return str(key or "").replace("_", " ").strip().title()


def _truncate(text: str, limit: int = 220) -> str:
    cleaned_text = _clean_text(text)
    if len(cleaned_text) <= limit:
        return cleaned_text
    shortened = cleaned_text[: max(0, limit - 3)].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return f"{shortened}..."


def _dedupe_text_items(items: Sequence[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []

    for item in items:
        cleaned_item = _clean_text(item)
        if not cleaned_item:
            continue

        normalized_item = cleaned_item.lower()
        if normalized_item in seen:
            continue

        seen.add(normalized_item)
        deduped.append(cleaned_item)

        if limit is not None and len(deduped) >= limit:
            break

    return deduped


def _sanitize_markdown_label(label: str) -> str:
    return _clean_text(label).replace("[", "(").replace("]", ")")


def _looks_like_follow_up(query: str) -> bool:
    cleaned_query = _clean_text(query).lower()
    return any(cleaned_query.startswith(prefix) for prefix in FOLLOW_UP_HINTS)


def _last_user_message(history: Sequence[Mapping[str, Any]] | None) -> str:
    if not history:
        return ""

    for entry in reversed(history):
        if not isinstance(entry, Mapping):
            continue

        role = _clean_text(str(entry.get("role") or "")).lower()
        if role == "user":
            return _clean_text(str(entry.get("content") or ""))

        user_message = _clean_text(
            str(
                entry.get("user_message")
                or entry.get("user_input")
                or entry.get("user")
                or ""
            )
        )
        if user_message:
            return user_message

    return ""


def build_live_web_query(
    user_message: str,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    cleaned_message = _clean_text(user_message)
    if not cleaned_message:
        raise ValueError("User message cannot be empty.")

    previous_user_message = _last_user_message(history)
    if not previous_user_message:
        return cleaned_message

    if not _looks_like_follow_up(cleaned_message):
        return cleaned_message

    normalized_previous = previous_user_message.lower()
    normalized_current = cleaned_message.lower()
    if normalized_previous in normalized_current:
        return cleaned_message

    return f"{previous_user_message} {cleaned_message}"


def _build_search_params(query: str) -> dict[str, str]:
    params = {
        "engine": SERPAPI_ENGINE,
        "q": query,
        "api_key": SERPAPI_API_KEY,
    }
    if SERPAPI_HL:
        params["hl"] = SERPAPI_HL
    if SERPAPI_GL:
        params["gl"] = SERPAPI_GL
    if SERPAPI_LOCATION:
        params["location"] = SERPAPI_LOCATION
    return params


def _extract_error_detail(response_text: str) -> str:
    try:
        payload = json.loads(response_text)
    except ValueError:
        return response_text.strip()

    if isinstance(payload, Mapping):
        for key in ("error", "message", "detail"):
            detail = payload.get(key)
            if detail:
                return _to_text(detail)

    return response.text.strip()


def _build_live_web_error(detail: str, status_code: int | None = None) -> LiveWebAPIError:
    cleaned_detail = _clean_text(detail)
    normalized_detail = cleaned_detail.lower()

    if status_code in {401, 403} or "invalid api key" in normalized_detail:
        return LiveWebAPIError(
            "Online mode is unavailable right now. The SerpAPI key was rejected. Check SERPAPI_API_KEY and try again.",
            category="invalid_key",
            status_code=status_code,
            detail=cleaned_detail,
        )

    if status_code == 429 or any(
        phrase in normalized_detail
        for phrase in (
            "rate limit",
            "quota",
            "searches left",
            "used all searches",
            "monthly searches limit",
            "plan searches are exhausted",
        )
    ):
        return LiveWebAPIError(
            "Online mode is temporarily unavailable. The SerpAPI quota or rate limit was reached.",
            category="rate_limited",
            status_code=status_code,
            detail=cleaned_detail,
        )

    if status_code == 404:
        return LiveWebAPIError(
            "Online mode is unavailable right now. The SerpAPI endpoint could not be reached.",
            category="not_found",
            status_code=status_code,
            detail=cleaned_detail,
        )

    if status_code == 408:
        return LiveWebAPIError(
            "Online mode is temporarily unavailable.",
            category="timeout",
            status_code=status_code,
            detail=cleaned_detail,
        )

    if cleaned_detail:
        return LiveWebAPIError(
            f"Online mode is unavailable right now. {cleaned_detail}",
            category="api_error",
            status_code=status_code,
            detail=cleaned_detail,
        )

    return LiveWebAPIError(
        "Online mode is temporarily unavailable.",
        category="api_error",
        status_code=status_code,
        detail=cleaned_detail,
    )


def _ensure_live_web_ready() -> None:
    if is_live_web_configured():
        return

    if not SERPAPI_API_KEY.strip():
        raise LiveWebConfigError(
            "Online mode is unavailable right now. Add SERPAPI_API_KEY to your environment and try again."
        )
    if not ENABLE_LIVE_WEB:
        raise LiveWebConfigError("Online mode is disabled. Set ENABLE_LIVE_WEB=true to turn it on.")
    if not SERPAPI_ENDPOINT.strip():
        raise LiveWebConfigError("Online mode is unavailable right now. The SerpAPI endpoint is missing.")
    if not SERPAPI_ENGINE.strip():
        raise LiveWebConfigError("Online mode is unavailable right now. The SerpAPI engine is missing.")


def _collect_mapping_facts(
    value: Mapping[str, Any] | None,
    preferred_keys: Sequence[str],
    limit: int = DETAIL_LIMIT,
) -> list[str]:
    if not isinstance(value, Mapping):
        return []

    facts: list[str] = []
    for key in preferred_keys:
        fact_value = _sanitize_live_web_fragment(_to_text(value.get(key)))
        if not fact_value:
            continue
        facts.append(f"{_friendly_label(key)}: {fact_value}")
        if len(facts) >= limit:
            break
    return facts


def _flatten_section_facts(
    value: Any,
    *,
    prefix: str = "",
    limit: int = DETAIL_LIMIT,
    depth: int = 0,
) -> list[str]:
    if limit <= 0 or depth > 3:
        return []

    if isinstance(value, Mapping):
        facts: list[str] = []
        for key, nested_value in value.items():
            if len(facts) >= limit:
                break
            nested_prefix = _friendly_label(key) if not prefix else f"{prefix} {_friendly_label(key)}"
            facts.extend(
                _flatten_section_facts(
                    nested_value,
                    prefix=nested_prefix,
                    limit=limit - len(facts),
                    depth=depth + 1,
                )
            )
        return facts

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        facts: list[str] = []
        for item in value:
            if len(facts) >= limit:
                break
            facts.extend(
                _flatten_section_facts(
                    item,
                    prefix=prefix,
                    limit=limit - len(facts),
                    depth=depth + 1,
                )
            )
        return facts

    scalar_value = _to_text(value)
    scalar_value = _sanitize_live_web_fragment(scalar_value)
    if not scalar_value:
        return []

    if prefix:
        return [f"{prefix}: {scalar_value}"]
    return [scalar_value]


def _summarize_answer_box(answer_box: Any) -> tuple[str, list[str]]:
    if not isinstance(answer_box, Mapping):
        return "", []

    answer_title = _to_text(answer_box.get("title"))
    direct_answer = _first_non_empty(
        answer_box.get("answer"),
        answer_box.get("snippet"),
        answer_box.get("result"),
        answer_box.get("snippet_highlighted_words"),
    )

    answer_title = _sanitize_live_web_fragment(answer_title)
    direct_answer = _sanitize_live_web_fragment(direct_answer)

    if answer_title and direct_answer and answer_title.lower() not in direct_answer.lower():
        direct_answer = f"{answer_title}: {direct_answer}"

    facts = _collect_mapping_facts(
        answer_box,
        (
            "answer",
            "result",
            "price",
            "exchange",
            "currency",
            "temperature",
            "high",
            "low",
            "humidity",
            "wind",
            "hours",
            "status",
        ),
    )
    return direct_answer, facts


def _summarize_weather_result(weather_result: Any) -> tuple[str, list[str]]:
    if not isinstance(weather_result, Mapping):
        return "", []

    title = _first_non_empty(weather_result.get("location"), weather_result.get("title"))
    temperature = _first_non_empty(
        weather_result.get("temperature"),
        weather_result.get("temp_f"),
        weather_result.get("temp_c"),
    )
    condition = _first_non_empty(weather_result.get("condition"), weather_result.get("weather"))

    direct_parts = [part for part in (title, temperature, condition) if part]
    direct_answer = ""
    if direct_parts:
        direct_answer = ": ".join(direct_parts[:2])
        if len(direct_parts) > 2:
            direct_answer = f"{direct_answer}, {direct_parts[2]}"
    direct_answer = _sanitize_live_web_fragment(direct_answer)

    facts = _collect_mapping_facts(
        weather_result,
        ("temperature", "condition", "high", "low", "humidity", "wind"),
    )
    return direct_answer, facts


def _summarize_knowledge_graph(knowledge_graph: Any) -> tuple[str, list[str]]:
    if not isinstance(knowledge_graph, Mapping):
        return "", []

    title = _to_text(knowledge_graph.get("title"))
    entity_type = _to_text(knowledge_graph.get("type"))
    description = _to_text(knowledge_graph.get("description"))

    direct_answer = _first_non_empty(description, title)
    if title and entity_type and entity_type.lower() not in direct_answer.lower():
        direct_answer = f"{title} ({entity_type}): {description or title}"
    direct_answer = _sanitize_live_web_fragment(direct_answer)

    facts = _collect_mapping_facts(
        knowledge_graph,
        ("type", "description", "born", "died", "headquarters", "founder", "website"),
    )
    return direct_answer, facts


def _summarize_sports_results(sports_results: Any) -> tuple[str, list[str]]:
    if not isinstance(sports_results, Mapping):
        return "", []

    direct_answer = _sanitize_live_web_fragment(
        _first_non_empty(sports_results.get("title"), sports_results.get("game_spotlight"))
    )
    facts = _flatten_section_facts(
        sports_results.get("game_spotlight") or sports_results.get("games") or sports_results,
        limit=DETAIL_LIMIT,
    )
    return direct_answer, facts


def _extract_result_snippet(result: Mapping[str, Any]) -> str:
    snippet = _first_non_empty(
        result.get("snippet"),
        result.get("snippet_highlighted_words"),
        result.get("description"),
    )
    if snippet:
        return snippet

    rich_snippet = result.get("rich_snippet")
    if isinstance(rich_snippet, Mapping):
        return _first_non_empty(
            rich_snippet.get("top"),
            rich_snippet.get("bottom"),
            rich_snippet.get("extensions"),
        )

    return ""


def _normalize_result_entry(result: Mapping[str, Any]) -> dict[str, str] | None:
    title = _sanitize_live_web_fragment(_to_text(result.get("title")))
    url = _first_non_empty(result.get("link"), result.get("url"))
    snippet = _sanitize_live_web_fragment(_extract_result_snippet(result))
    source = _sanitize_live_web_fragment(_first_non_empty(result.get("source"), result.get("displayed_link")))
    if not source:
        source = _domain_from_url(url)

    result_date = _to_text(result.get("date"))
    if result_date and snippet and result_date.lower() not in snippet.lower():
        snippet = f"{result_date}. {snippet}"
        snippet = _sanitize_live_web_fragment(snippet)

    if not title and not snippet:
        return None

    return {
        "title": title,
        "snippet": _truncate(snippet, 240),
        "url": url,
        "source": source,
    }


def normalize_search_results(payload: Mapping[str, Any], query: str) -> dict[str, Any]:
    direct_answer = ""
    facts: list[str] = []

    for summarizer, section_name in (
        (_summarize_answer_box, "answer_box"),
        (_summarize_weather_result, "weather_result"),
        (_summarize_sports_results, "sports_results"),
        (_summarize_knowledge_graph, "knowledge_graph"),
    ):
        section = payload.get(section_name)
        section_answer, section_facts = summarizer(section)
        if section_answer and not direct_answer:
            direct_answer = section_answer
        facts.extend(section_facts)

    normalized_results: list[dict[str, str]] = []
    seen_result_keys: set[tuple[str, str]] = set()

    for section_name in ("top_stories", "news_results", "organic_results"):
        section = payload.get(section_name)
        if not isinstance(section, Sequence) or isinstance(section, (str, bytes, bytearray)):
            continue

        for item in section:
            if not isinstance(item, Mapping):
                continue

            normalized_result = _normalize_result_entry(item)
            if not normalized_result:
                continue

            unique_key = (
                normalized_result["url"].lower(),
                normalized_result["title"].lower(),
            )
            if unique_key in seen_result_keys:
                continue

            seen_result_keys.add(unique_key)
            normalized_results.append(normalized_result)

            if len(normalized_results) >= RESULT_LIMIT:
                break

        if len(normalized_results) >= RESULT_LIMIT:
            break

    if not direct_answer and normalized_results:
        top_result = normalized_results[0]
        direct_answer = _first_non_empty(top_result.get("snippet"), top_result.get("title"))
    direct_answer = _sanitize_live_web_fragment(direct_answer)

    facts = _dedupe_text_items(facts, limit=DETAIL_LIMIT)
    if direct_answer:
        facts = [
            fact
            for fact in facts
            if _clean_text(fact).lower() != _clean_text(direct_answer).lower()
        ]
    facts = [fact for fact in (_sanitize_live_web_fragment(fact) for fact in facts) if fact]

    return {
        "query": _clean_text(query),
        "direct_answer": _truncate(direct_answer, 260),
        "facts": facts,
        "results": normalized_results,
    }


def _normalize_query_type(query_type: str | None) -> str:
    normalized_query_type = _clean_text(query_type).lower().replace(" ", "_")
    if normalized_query_type in {QUERY_TYPE_WEB_LOOKUP, "web", "lookup"}:
        return QUERY_TYPE_WEB_LOOKUP
    if normalized_query_type in {QUERY_TYPE_HYBRID_TASK, "hybrid"}:
        return QUERY_TYPE_HYBRID_TASK
    if normalized_query_type in {QUERY_TYPE_GENERATIVE_TASK, "generative"}:
        return QUERY_TYPE_GENERATIVE_TASK
    return QUERY_TYPE_WEB_LOOKUP


def _ensure_normalized_search_data(query: str, serp_result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(serp_result, Mapping):
        return {
            "query": _clean_text(query),
            "direct_answer": "",
            "facts": [],
            "results": [],
        }

    if {"direct_answer", "facts", "results"}.issubset(set(serp_result.keys())):
        normalized_data = dict(serp_result)
        normalized_data["query"] = _clean_text(str(normalized_data.get("query") or query))
        normalized_data["direct_answer"] = _clean_text(str(normalized_data.get("direct_answer") or ""))
        normalized_data["facts"] = list(normalized_data.get("facts") or [])
        normalized_data["results"] = list(normalized_data.get("results") or [])
        return normalized_data

    return normalize_search_results(serp_result, query=query)


def _build_detail_points(search_data: Mapping[str, Any]) -> list[str]:
    detail_points: list[str] = []

    for fact in search_data.get("facts", []):
        sanitized_fact = _sanitize_live_web_fragment(_to_text(fact))
        if sanitized_fact:
            detail_points.append(_truncate(sanitized_fact, 200))

    results = search_data.get("results", [])
    if isinstance(results, Sequence):
        for result in results:
            if not isinstance(result, Mapping):
                continue

            title = _to_text(result.get("title"))
            snippet = _to_text(result.get("snippet"))
            source = _to_text(result.get("source"))

            if title and snippet:
                point = f"{title}: {snippet}"
            else:
                point = title or snippet

            if source:
                point = f"{point} ({source})"

            point = _sanitize_live_web_fragment(point)
            if point:
                detail_points.append(_truncate(point, 220))

            if len(detail_points) >= DETAIL_LIMIT:
                break

    return _dedupe_text_items(detail_points, limit=DETAIL_LIMIT)


def _web_label(language_context: LanguageContext | None, key: str) -> str:
    code = language_context.code if language_context is not None else "en"
    labels = WEB_LABELS.get(code, WEB_LABELS["en"])
    return labels.get(key, WEB_LABELS["en"][key])


def _build_sources_block(
    results: Sequence[Mapping[str, Any]] | None,
    language_context: LanguageContext | None = None,
) -> str:
    if not results:
        return ""

    lines = [_web_label(language_context, "sources")]
    count = 0

    for result in results:
        if not isinstance(result, Mapping):
            continue

        url = _to_text(result.get("url"))
        if not url:
            continue

        title = _sanitize_markdown_label(
            _first_non_empty(result.get("title"), result.get("source"), url)
        )
        source = _sanitize_live_web_fragment(_to_text(result.get("source")))
        suffix = f" - {source}" if source else ""
        lines.append(f"- [{title}]({url}){suffix}")
        count += 1

        if count >= SOURCE_LIMIT:
            break

    if count == 0:
        return ""

    return "\n".join(lines)


def build_web_grounded_answer(
    query: str,
    serp_result: Mapping[str, Any],
    query_type: str,
    language_context: LanguageContext | None = None,
) -> str:
    cleaned_query = _clean_text(query)
    normalized_query_type = _normalize_query_type(query_type)
    search_data = _ensure_normalized_search_data(cleaned_query, serp_result)

    summary = _sanitize_live_web_fragment(_to_text(search_data.get("direct_answer")))
    detail_points = _build_detail_points(search_data)
    sources_block = _build_sources_block(search_data.get("results"), language_context=language_context)

    if not summary:
        results = search_data.get("results", [])
        if isinstance(results, Sequence):
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                summary = _sanitize_live_web_fragment(
                    _first_non_empty(
                        result.get("snippet"),
                        result.get("title"),
                        result.get("source"),
                    )
                )
                if summary:
                    break

    if not summary:
        summary = _web_label(language_context, "no_clear_answer")

    parts = [summary]

    if detail_points:
        parts.append(_web_label(language_context, "quick_details") + "\n" + "\n".join(f"- {point}" for point in detail_points))
    elif normalized_query_type == QUERY_TYPE_HYBRID_TASK:
        parts.append(
            _web_label(language_context, "quick_details")
            + "\n- "
            + _web_label(language_context, "limited_detail")
        )

    if normalized_query_type == QUERY_TYPE_HYBRID_TASK:
        parts.append(_web_label(language_context, "hybrid_followup"))

    if sources_block:
        parts.append(sources_block)

    return "\n\n".join(parts)


def build_live_web_reply(
    query: str,
    mode: str,
    results: Mapping[str, Any],
    language_context: LanguageContext | None = None,
) -> str:
    del mode
    return build_web_grounded_answer(
        query,
        results,
        query_type=classify_query(query),
        language_context=language_context,
    )


def compose_live_web_answer(
    search_data: Mapping[str, Any],
    mode: str = "playful",
    language_context: LanguageContext | None = None,
) -> str:
    query = _to_text(search_data.get("query"))
    del mode
    return build_web_grounded_answer(
        query,
        search_data,
        query_type=QUERY_TYPE_WEB_LOOKUP,
        language_context=language_context,
    )


def search_live_web(
    user_message: str,
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    _ensure_live_web_ready()
    query = build_live_web_query(user_message, history=history)
    params = _build_search_params(query)
    request_timeout = timeout if timeout is not None else SERPAPI_REQUEST_TIMEOUT_SECONDS

    logger.debug(
        "Online request prepared endpoint=%s engine=%s key=%s query=%s",
        SERPAPI_ENDPOINT,
        SERPAPI_ENGINE,
        mask_secret(SERPAPI_API_KEY),
        query,
    )

    request_url = f"{SERPAPI_ENDPOINT}?{urllib_parse.urlencode(params)}"
    request = urllib_request.Request(
        request_url,
        headers={
            "Accept": "application/json",
        },
        method="GET",
    )
    opener = _build_http_session()
    response = None
    try:
        if hasattr(opener, "open"):
            response = opener.open(request, timeout=request_timeout)
        else:
            response = opener.get(request, timeout=request_timeout)
    except urllib_error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace").strip()
        detail = _extract_error_detail(response_text)
        raise _build_live_web_error(detail, status_code=exc.code) from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise LiveWebAPIError(
            "Online mode is temporarily unavailable.",
            category="timeout",
            detail=_clean_text(str(exc)),
        ) from exc
    finally:
        if hasattr(opener, "close"):
            opener.close()

    response_status = getattr(response, "status", None)
    if response_status is None:
        response_status = response.getcode() if hasattr(response, "getcode") else None

    if hasattr(response, "read"):
        response_text = response.read().decode("utf-8", errors="replace").strip()
    elif hasattr(response, "json"):
        try:
            response_text = json.dumps(response.json())
        except Exception:
            response_text = ""
    else:
        response_text = ""

    if hasattr(response, "close"):
        response.close()

    if response_status and response_status >= 400:
        detail = _extract_error_detail(response_text)
        raise _build_live_web_error(detail, status_code=response_status)

    try:
        payload = json.loads(response_text)
    except ValueError as exc:
        raise LiveWebAPIError(
            "Online mode returned malformed search data.",
            category="malformed_response",
            detail=_clean_text(str(exc)),
        ) from exc

    if not payload:
        raise LiveWebAPIError(
            "Online mode returned an empty search response.",
            category="empty_response",
        )

    if not isinstance(payload, Mapping):
        raise LiveWebAPIError(
            "Online mode returned malformed search data.",
            category="malformed_response",
            detail=f"Unexpected response type: {type(payload).__name__}",
        )

    payload_error = _first_non_empty(payload.get("error"), payload.get("message"), payload.get("detail"))
    if payload_error:
        raise _build_live_web_error(payload_error, status_code=response.status_code)

    normalized_search = normalize_search_results(payload, query=query)
    if not normalized_search["direct_answer"] and not normalized_search["results"]:
        raise LiveWebAPIError(
            "I couldn't find fresh results for that query.",
            category="no_results",
        )

    return normalized_search


def generate_live_web_response(
    user_message: str,
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
    timeout: int | None = None,
    stream_callback: Callable[[str], None] | None = None,
    language_context: LanguageContext | None = None,
) -> str:
    resolved_language_context = language_context or resolve_language_context(user_message, history)
    search_data = search_live_web(
        user_message=user_message,
        history=history,
        timeout=timeout,
    )
    query = _to_text(search_data.get("query")) or _clean_text(user_message)
    query_type = classify_query(user_message)
    reply_text = build_web_grounded_answer(
        query,
        search_data,
        query_type=query_type,
        language_context=resolved_language_context,
    )
    cleaned_reply = sanitize_live_web_output(reply_text, query=query)
    if not cleaned_reply:
        cleaned_reply = sanitize_live_web_output(
            _web_label(resolved_language_context, "no_clear_answer"),
            query=query,
        )
    if not cleaned_reply:
        raise LiveWebAPIError(
            "I couldn't find fresh results for that query.",
            category="no_results",
        )

    if stream_callback is not None and cleaned_reply:
        stream_callback(cleaned_reply)

    return cleaned_reply


def get_live_web_debug_snapshot() -> dict[str, Any]:
    return {
        "online_enabled": ENABLE_LIVE_WEB,
        "api_key_present": bool(SERPAPI_API_KEY.strip()),
        "api_key_length": len(SERPAPI_API_KEY.strip()),
        "api_key_masked": mask_secret(SERPAPI_API_KEY),
        "endpoint": SERPAPI_ENDPOINT,
        "engine": SERPAPI_ENGINE,
        "hl": SERPAPI_HL,
        "gl": SERPAPI_GL,
        "location": SERPAPI_LOCATION,
        "trust_env_proxy": SERPAPI_TRUST_ENV_PROXY,
    }


def debug_live_web_connection(
    user_message: str = "latest ai news",
    timeout: int | None = None,
) -> dict[str, Any]:
    snapshot = get_live_web_debug_snapshot()

    try:
        search_data = search_live_web(
            user_message=user_message,
            history=[],
            timeout=timeout,
        )
        reply_text = compose_live_web_answer(search_data, mode="playful")
    except LiveWebConfigError as exc:
        snapshot.update(
            ok=False,
            category="missing_config",
            message=str(exc),
        )
        return snapshot
    except LiveWebAPIError as exc:
        snapshot.update(
            ok=False,
            category=exc.category,
            status_code=exc.status_code,
            message=str(exc),
            detail=exc.detail,
        )
        return snapshot

    snapshot.update(
        ok=True,
        category="ok",
        message="Online mode check succeeded.",
        result_count=len(search_data.get("results", [])),
        answer_preview=reply_text[:200],
    )
    return snapshot
