from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from config import (
    DEFAULT_MODE,
    DEFAULT_MODEL_NAME,
    DEFAULT_RUNTIME_PREFERENCE,
    OFFLINE_OLLAMA_MODEL,
)
from core.chatbot import OllamaError, generate_response
from core.language import (
    LanguageContext,
    build_language_instruction,
    language_metadata,
    resolve_language_context,
)
from core.live_web import (
    build_web_grounded_answer,
    LiveWebAPIError,
    LiveWebConfigError,
    is_live_web_configured,
    search_live_web,
)
from core.prompts import build_chat_prompt, build_summary_prompt, get_system_prompt, sanitize_identity_leaks
from core.router import (
    QUERY_TYPE_GENERATIVE_TASK,
    QUERY_TYPE_HYBRID_TASK,
    QUERY_TYPE_WEB_LOOKUP,
    classify_query,
)
# Code mode import removed
from features.prompt_gen import build_prompt_generation_prompt


class SonicProviderError(RuntimeError):
    pass


ONLINE_FALLBACK_NOTICE = (
    "Online mode is unavailable right now, so I'm answering from offline mode for this reply."
)


IDENTITY_PATTERNS = (
    r"\bwhich model are you\b",
    r"\bwhat model are you\b",
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat provider are you using\b",
    r"\bwho made you\b",
    r"\bare you nvidia\b",
    r"\bare you nemotron\b",
    r"\bcreated by nvidia\b",
    r"\bpowered by nvidia\b",
    r"\bare you serpapi\b",
    r"\bcreated by serpapi\b",
)


@dataclass(frozen=True, slots=True)
class PromptBundle:
    mode: str
    system_prompt: str
    offline_prompt: str
    live_web_history: list[dict[str, str]]
    live_web_query: str
    language_context: LanguageContext


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    runtime_used: str
    provider_used: str


def _build_provider_response(
    reply_text: str,
    *,
    runtime_used: str,
    provider_used: str,
    model_used: str,
    mode: str,
    notice: str = "",
    requested_runtime: str = "",
    fallback_used: bool = False,
    fallback_reason: str = "",
    language_context: LanguageContext | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "reply": reply_text,
        "runtime_used": runtime_used,
        "provider_used": provider_used,
        "model_used": model_used,
        "mode": mode,
    }
    if notice:
        response["notice"] = notice
    if requested_runtime:
        response["requested_runtime"] = requested_runtime
    if fallback_used:
        response["fallback_used"] = True
    if fallback_reason:
        response["fallback_reason"] = fallback_reason
    if language_context is not None:
        response.update(language_metadata(language_context))
    return response


def _clean_text(text: str) -> str:
    return (text or "").strip()


def _finalize_reply_text(reply_text: str, fallback_reply: str) -> str:
    cleaned_reply = sanitize_identity_leaks(reply_text).strip()
    if cleaned_reply:
        return cleaned_reply
    return fallback_reply


LOCAL_LIMIT_COPY = {
    "en": {
        "hybrid": 'I\'m SONIC. I\'m in Local mode right now, so I can\'t verify the live part of "{message}" directly. Switch to Online mode and I can check the current facts.',
        "hybrid_empty": "I'm SONIC. I'm in Local mode right now, so I can't verify live facts directly. Switch to Online mode and I can check them for you.",
        "lookup": 'I\'m SONIC. I\'m in Local mode right now, so I can\'t verify the live part of "{message}" directly. Switch to Online mode and I can check the current facts.',
        "lookup_empty": "I'm SONIC. I'm in Local mode right now, so I can't verify live facts directly. Switch to Online mode and I can check them for you.",
        "online_identity": "I'm SONIC. In Online mode I can use live web information through SerpAPI to pull current facts and turn them into grounded answers.",
        "local_identity": "I'm SONIC. In Local mode I'm running through the offline Ollama setup for private, offline-first chat.",
    },
    "hi": {
        "hybrid": 'Main SONIC hoon. Abhi Offline mode me hoon, isliye "{message}" ka live/current part directly verify nahi kar sakta. Online mode par switch karo, main current facts check kar dunga.',
        "hybrid_empty": "Main SONIC hoon. Abhi Offline mode me hoon, isliye live facts directly verify nahi kar sakta. Online mode par switch karo, main check kar dunga.",
        "lookup": 'Main SONIC hoon. Abhi Offline mode me hoon, isliye "{message}" ka live/current part directly verify nahi kar sakta. Online mode par switch karo, main current facts check kar dunga.',
        "lookup_empty": "Main SONIC hoon. Abhi Offline mode me hoon, isliye live facts directly verify nahi kar sakta. Online mode par switch karo, main check kar dunga.",
        "online_identity": "Main SONIC hoon. Online mode me main SerpAPI ke through live web info use karke current facts ko grounded answer me badal sakta hoon.",
        "local_identity": "Main SONIC hoon. Offline mode me main private, offline-first chat ke liye offline Ollama setup se chal raha hoon.",
    },
    "zh": {
        "hybrid": '我是 SONIC。现在是离线模式，所以不能直接核实“{message}”里的实时信息。切换到在线模式后，我可以检查最新事实。',
        "hybrid_empty": "我是 SONIC。现在是离线模式，所以不能直接核实实时事实。切换到在线模式后我可以帮你检查。",
        "lookup": '我是 SONIC。现在是离线模式，所以不能直接核实“{message}”里的实时信息。切换到在线模式后，我可以检查最新事实。',
        "lookup_empty": "我是 SONIC。现在是离线模式，所以不能直接核实实时事实。切换到在线模式后我可以帮你检查。",
        "online_identity": "我是 SONIC。在线模式下，我可以通过 SerpAPI 使用实时网页信息，并整理成有依据的回答。",
        "local_identity": "我是 SONIC。离线模式下，我通过本机 Ollama 设置运行，用于私密、离线优先的聊天。",
    },
    "es": {
        "hybrid": 'Soy SONIC. Ahora estoy en modo offline, así que no puedo verificar directamente la parte en vivo de "{message}". Cambia a modo online y reviso los datos actuales.',
        "hybrid_empty": "Soy SONIC. Ahora estoy en modo offline, así que no puedo verificar datos en vivo directamente. Cambia a modo online y los reviso.",
        "lookup": 'Soy SONIC. Ahora estoy en modo offline, así que no puedo verificar directamente la parte en vivo de "{message}". Cambia a modo online y reviso los datos actuales.',
        "lookup_empty": "Soy SONIC. Ahora estoy en modo offline, así que no puedo verificar datos en vivo directamente. Cambia a modo online y los reviso.",
        "online_identity": "Soy SONIC. En modo online puedo usar información web en vivo mediante SerpAPI para obtener datos actuales y convertirlos en respuestas fundamentadas.",
        "local_identity": "Soy SONIC. En modo offline funciono con la configuración offline de Ollama para chat privado y offline-first.",
    },
    "fr": {
        "hybrid": 'Je suis SONIC. Je suis en mode offline pour l’instant, donc je ne peux pas vérifier directement la partie en direct de "{message}". Passe en mode online et je pourrai vérifier les faits actuels.',
        "hybrid_empty": "Je suis SONIC. Je suis en mode offline pour l’instant, donc je ne peux pas vérifier directement les faits en direct. Passe en mode online et je pourrai les vérifier.",
        "lookup": 'Je suis SONIC. Je suis en mode offline pour l’instant, donc je ne peux pas vérifier directement la partie en direct de "{message}". Passe en mode online et je pourrai vérifier les faits actuels.',
        "lookup_empty": "Je suis SONIC. Je suis en mode offline pour l’instant, donc je ne peux pas vérifier directement les faits en direct. Passe en mode online et je pourrai les vérifier.",
        "online_identity": "Je suis SONIC. En mode online, je peux utiliser les informations web en direct via SerpAPI pour récupérer des faits actuels et produire des réponses sourcées.",
        "local_identity": "Je suis SONIC. En mode offline, je fonctionne avec la configuration Ollama offline pour un chat privé et offline-first.",
    },
    "bn": {
        "hybrid": 'আমি SONIC। এখন Offline mode-এ আছি, তাই "{message}"-এর live/current অংশ সরাসরি যাচাই করতে পারছি না। Online mode-এ গেলে আমি সাম্প্রতিক তথ্য দেখে বলতে পারব।',
        "hybrid_empty": "আমি SONIC। এখন Offline mode-এ আছি, তাই live facts সরাসরি যাচাই করতে পারছি না। Online mode-এ গেলে আমি দেখে বলতে পারব।",
        "lookup": 'আমি SONIC। এখন Offline mode-এ আছি, তাই "{message}"-এর live/current অংশ সরাসরি যাচাই করতে পারছি না। Online mode-এ গেলে আমি সাম্প্রতিক তথ্য দেখে বলতে পারব।',
        "lookup_empty": "আমি SONIC। এখন Offline mode-এ আছি, তাই live facts সরাসরি যাচাই করতে পারছি না। Online mode-এ গেলে আমি দেখে বলতে পারব।",
        "online_identity": "আমি SONIC। Online mode-এ আমি SerpAPI দিয়ে live web information ব্যবহার করে current facts-কে grounded answer বানাতে পারি।",
        "local_identity": "আমি SONIC। Offline mode-এ আমি private, offline-first chat-এর জন্য offline Ollama setup দিয়ে চলি।",
    },
    "te": {
        "hybrid": 'నేను SONIC. ఇప్పుడు Offline mode‌లో ఉన్నాను, కాబట్టి "{message}" లోని live/current భాగాన్ని నేరుగా verify చేయలేను. Online mode‌కు మారితే తాజా facts చూసి చెబుతాను.',
        "hybrid_empty": "నేను SONIC. ఇప్పుడు Offline mode‌లో ఉన్నాను, కాబట్టి live facts‌ను నేరుగా verify చేయలేను. Online mode‌కు మారితే చూసి చెబుతాను.",
        "lookup": 'నేను SONIC. ఇప్పుడు Offline mode‌లో ఉన్నాను, కాబట్టి "{message}" లోని live/current భాగాన్ని నేరుగా verify చేయలేను. Online mode‌కు మారితే తాజా facts చూసి చెబుతాను.',
        "lookup_empty": "నేను SONIC. ఇప్పుడు Offline mode‌లో ఉన్నాను, కాబట్టి live facts‌ను నేరుగా verify చేయలేను. Online mode‌కు మారితే చూసి చెబుతాను.",
        "online_identity": "నేను SONIC. Online mode‌లో SerpAPI ద్వారా live web information తీసుకుని current facts‌ను grounded answers‌గా మార్చగలను.",
        "local_identity": "నేను SONIC. Offline mode‌లో private, offline-first chat కోసం offline Ollama setup‌తో నడుస్తాను.",
    },
    "mr": {
        "hybrid": 'मी SONIC आहे. आत्ता Offline mode मध्ये आहे, त्यामुळे "{message}" मधला live/current भाग थेट पडताळू शकत नाही. Online mode वर गेलात तर मी अद्ययावत facts तपासू शकतो.',
        "hybrid_empty": "मी SONIC आहे. आत्ता Offline mode मध्ये आहे, त्यामुळे live facts थेट पडताळू शकत नाही. Online mode वर गेलात तर मी तपासू शकतो.",
        "lookup": 'मी SONIC आहे. आत्ता Offline mode मध्ये आहे, त्यामुळे "{message}" मधला live/current भाग थेट पडताळू शकत नाही. Online mode वर गेलात तर मी अद्ययावत facts तपासू शकतो.',
        "lookup_empty": "मी SONIC आहे. आत्ता Offline mode मध्ये आहे, त्यामुळे live facts थेट पडताळू शकत नाही. Online mode वर गेलात तर मी तपासू शकतो.",
        "online_identity": "मी SONIC आहे. Online mode मध्ये मी SerpAPI द्वारे live web information वापरून current facts चे grounded answers देऊ शकतो.",
        "local_identity": "मी SONIC आहे. Offline mode मध्ये private, offline-first chat साठी offline Ollama setup वर चालतो.",
    },
    "ru": {
        "hybrid": 'Я SONIC. Сейчас я в offline mode, поэтому не могу напрямую проверить актуальную часть запроса "{message}". Переключись в online mode, и я проверю текущие факты.',
        "hybrid_empty": "Я SONIC. Сейчас я в offline mode, поэтому не могу напрямую проверять live-факты. Переключись в online mode, и я проверю их.",
        "lookup": 'Я SONIC. Сейчас я в offline mode, поэтому не могу напрямую проверить актуальную часть запроса "{message}". Переключись в online mode, и я проверю текущие факты.',
        "lookup_empty": "Я SONIC. Сейчас я в offline mode, поэтому не могу напрямую проверять live-факты. Переключись в online mode, и я проверю их.",
        "online_identity": "Я SONIC. В online mode я могу использовать live-информацию из веба через SerpAPI, чтобы получать актуальные факты и оформлять обоснованные ответы.",
        "local_identity": "Я SONIC. В offline mode я работаю через offline настройку Ollama для приватного, offline-first чата.",
    },
    "gu": {
        "hybrid": 'હું SONIC છું. હમણાં Offline mode માં છું, એટલે "{message}" નો live/current ભાગ સીધો verify કરી શકતો નથી. Online mode પર સ્વિચ કરો, પછી હું તાજા facts ચેક કરી શકું.',
        "hybrid_empty": "હું SONIC છું. હમણાં Offline mode માં છું, એટલે live facts સીધા verify કરી શકતો નથી. Online mode પર સ્વિચ કરો, પછી હું ચેક કરી શકું.",
        "lookup": 'હું SONIC છું. હમણાં Offline mode માં છું, એટલે "{message}" નો live/current ભાગ સીધો verify કરી શકતો નથી. Online mode પર સ્વિચ કરો, પછી હું તાજા facts ચેક કરી શકું.',
        "lookup_empty": "હું SONIC છું. હમણાં Offline mode માં છું, એટલે live facts સીધા verify કરી શકતો નથી. Online mode પર સ્વિચ કરો, પછી હું ચેક કરી શકું.",
        "online_identity": "હું SONIC છું. Online mode માં હું SerpAPI દ્વારા live web information લઈને current facts ને grounded answers માં ફેરવી શકું.",
        "local_identity": "હું SONIC છું. Offline mode માં હું private, offline-first chat માટે offline Ollama setup પરથી ચાલું છું.",
    },
}


def _localized_copy(language_context: LanguageContext | None, key: str) -> str:
    code = language_context.code if language_context is not None else "en"
    return LOCAL_LIMIT_COPY.get(code, LOCAL_LIMIT_COPY["en"]).get(key, LOCAL_LIMIT_COPY["en"][key])


def _build_offline_limit_reply(
    query_type: str,
    user_message: str,
    language_context: LanguageContext | None = None,
) -> str:
    normalized_query_type = _clean_text(query_type).lower()
    if normalized_query_type == QUERY_TYPE_GENERATIVE_TASK:
        return ""

    cleaned_message = _clean_text(user_message)
    if normalized_query_type == QUERY_TYPE_HYBRID_TASK:
        if cleaned_message:
            return _localized_copy(language_context, "hybrid").format(message=cleaned_message)
        return _localized_copy(language_context, "hybrid_empty")

    if cleaned_message:
        return _localized_copy(language_context, "lookup").format(message=cleaned_message)

    return _localized_copy(language_context, "lookup_empty")


def _is_identity_query(text: str) -> bool:
    cleaned_text = _clean_text(text).lower()
    if not cleaned_text:
        return False
    return any(re.search(pattern, cleaned_text) for pattern in IDENTITY_PATTERNS)


def _build_runtime_identity_reply(
    runtime_used: str,
    language_context: LanguageContext | None = None,
) -> str:
    normalized_runtime = _clean_text(runtime_used).lower()
    if normalized_runtime == "online":
        return _localized_copy(language_context, "online_identity")

    return _localized_copy(language_context, "local_identity")


def _normalize_mode(mode: str | None) -> str:
    normalized_mode = _clean_text(mode).lower().lstrip("/")
    if normalized_mode in {"prompt", "summary"}:
        return normalized_mode
    return "assistant"


def _normalize_runtime_preference(runtime_preference: str | None) -> str:
    normalized_runtime = _clean_text(runtime_preference).lower().lstrip("/")
    if normalized_runtime in {"offline", "local"}:
        return "offline"
    if normalized_runtime in {"online", "live", "liveweb", "live_web", "web"}:
        return "online"
    if normalized_runtime == "auto":
        return "auto"
    return DEFAULT_RUNTIME_PREFERENCE


def _normalize_history(
    history: Sequence[Any] | None,
) -> list[dict[str, str]]:
    if not history:
        return []

    normalized_history: list[dict[str, str]] = []
    pending_user_message: str | None = None

    for item in history:
        if isinstance(item, Mapping):
            if "role" in item and "content" in item:
                role = _clean_text(str(item.get("role") or "")).lower()
                content = _clean_text(str(item.get("content") or ""))
                if not content:
                    continue

                if role == "user":
                    pending_user_message = content
                elif role == "assistant" and pending_user_message is not None:
                    normalized_history.append(
                        {
                            "user_message": pending_user_message,
                            "assistant_response": content,
                        }
                    )
                    pending_user_message = None
                continue

            user_message = _clean_text(
                str(
                    item.get("user_message")
                    or item.get("user_input")
                    or item.get("user")
                    or ""
                )
            )
            assistant_response = _clean_text(
                str(
                    item.get("assistant_response")
                    or item.get("assistant_reply")
                    or item.get("assistant")
                    or ""
                )
            )
            if user_message or assistant_response:
                normalized_history.append(
                    {
                        "user_message": user_message,
                        "assistant_response": assistant_response,
                    }
                )
            continue

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) < 2:
                continue

            user_message = _clean_text(str(item[0]))
            assistant_response = _clean_text(str(item[1]))
            if user_message or assistant_response:
                normalized_history.append(
                    {
                        "user_message": user_message,
                        "assistant_response": assistant_response,
                    }
                )

    return normalized_history


def _history_to_live_web_messages(history: Sequence[Mapping[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    live_web_messages: list[dict[str, str]] = []
    for entry in history:
        user_message = _clean_text(str(entry.get("user_message") or ""))
        assistant_response = _clean_text(str(entry.get("assistant_response") or ""))

        if user_message:
            live_web_messages.append({"role": "user", "content": user_message})
        if assistant_response:
            live_web_messages.append({"role": "assistant", "content": assistant_response})

    return live_web_messages


def _build_prompt_bundle(
    message: str,
    mode: str,
    history: Sequence[Any] | None,
) -> PromptBundle:
    normalized_mode = _normalize_mode(mode)
    cleaned_message = _clean_text(message)
    normalized_history = _normalize_history(history)
    language_context = resolve_language_context(cleaned_message, history)
    live_web_history = _history_to_live_web_messages(normalized_history)

    if normalized_mode == "summary":
        summary_prompt = build_summary_prompt(
            normalized_history,
            language_context=language_context,
        )
        return PromptBundle(
            mode=normalized_mode,
            system_prompt=get_system_prompt("summary", language_context=language_context),
            offline_prompt=summary_prompt,
            live_web_history=[],
            live_web_query="",
            language_context=language_context,
        )

    if normalized_mode == "prompt":
        prompt_generation_prompt = build_prompt_generation_prompt(cleaned_message, language_context=language_context)
        return PromptBundle(
            mode=normalized_mode,
            system_prompt=get_system_prompt("prompt", language_context=language_context),
            offline_prompt=prompt_generation_prompt,
            live_web_history=live_web_history,
            live_web_query=cleaned_message,
            language_context=language_context,
        )

    return PromptBundle(
        mode=normalized_mode,
        system_prompt=get_system_prompt(normalized_mode, language_context=language_context),
        offline_prompt=build_chat_prompt(
            cleaned_message,
            normalized_history,
            language_context=language_context,
        ),
        live_web_history=live_web_history,
        live_web_query=cleaned_message,
        language_context=language_context,
    )


def _resolve_runtime_selection(
    runtime_preference: str | None,
) -> RuntimeSelection:
    normalized_runtime = _normalize_runtime_preference(runtime_preference)

    if normalized_runtime == "offline":
        return RuntimeSelection(
            runtime_used="local",
            provider_used="local",
        )

    if normalized_runtime == "online" or (
        normalized_runtime == "auto" and is_live_web_configured()
    ):
        return RuntimeSelection(
            runtime_used="online",
            provider_used="serpapi",
        )

    return RuntimeSelection(
        runtime_used="local",
        provider_used="local",
    )


def _run_local_provider(
    prompt_text: str,
    system_prompt: str,
    mode: str,
    runtime_used: str = "local",
    user_message: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
    language_context: LanguageContext | None = None,
) -> dict[str, Any]:
    offline_model = OFFLINE_OLLAMA_MODEL or DEFAULT_MODEL_NAME
    resolved_user_message = _clean_text(user_message or prompt_text)
    query_type = classify_query(resolved_user_message or prompt_text)
    offline_notice = _build_offline_limit_reply(query_type, resolved_user_message, language_context=language_context)
    if offline_notice:
        reply_text = _finalize_reply_text(
            offline_notice,
            _build_runtime_identity_reply(runtime_used, language_context=language_context),
        )
        if stream_callback is not None:
            stream_callback(reply_text)
        return _build_provider_response(
            reply_text,
            runtime_used=runtime_used,
            provider_used="local",
            model_used=offline_model,
            mode=mode,
            language_context=language_context,
        )

    if _is_identity_query(resolved_user_message or prompt_text):
        reply_text = _build_runtime_identity_reply(runtime_used, language_context=language_context)
        if stream_callback is not None:
            stream_callback(reply_text)
        return _build_provider_response(
            reply_text,
            runtime_used=runtime_used,
            provider_used="local",
            model_used=offline_model,
            mode=mode,
            language_context=language_context,
        )

    reply_text = generate_response(
        prompt_text,
        system_prompt=system_prompt,
        model=offline_model,
        mode=mode,
        stream_callback=stream_callback,
        prefer_stream=stream_callback is not None,
    )
    reply_text = _finalize_reply_text(
        reply_text,
        _build_runtime_identity_reply(runtime_used, language_context=language_context),
    )

    return _build_provider_response(
        reply_text,
        runtime_used=runtime_used,
        provider_used="local",
        model_used=offline_model,
        mode=mode,
        language_context=language_context,
    )


def _split_sources_block(reply_text: str) -> tuple[str, str]:
    cleaned_reply = (reply_text or "").strip()
    if not cleaned_reply:
        return "", ""

    marker = "\n\nSources:"
    marker_index = cleaned_reply.find(marker)
    if marker_index < 0:
        return cleaned_reply, ""

    return cleaned_reply[:marker_index].strip(), cleaned_reply[marker_index + 2 :].strip()


def _build_grounded_composer_prompt(
    *,
    query: str,
    search_data: Mapping[str, Any],
    draft_answer: str,
    language_context: LanguageContext,
) -> str:
    facts = search_data.get("facts", [])
    results = search_data.get("results", [])

    fact_lines: list[str] = []
    if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes, bytearray)):
        for fact in facts[:4]:
            fact_text = _clean_text(str(fact))
            if fact_text:
                fact_lines.append(f"- {fact_text}")

    result_lines: list[str] = []
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        for result in results[:4]:
            if not isinstance(result, Mapping):
                continue
            title = _clean_text(str(result.get("title") or result.get("source") or "Source"))
            snippet = _clean_text(str(result.get("snippet") or ""))
            source = _clean_text(str(result.get("source") or ""))
            line = f"- {title}"
            if source:
                line += f" ({source})"
            if snippet:
                line += f": {snippet}"
            result_lines.append(line)

    return "\n\n".join(
        [
            "You are SONIC composing a live-web grounded answer.",
            build_language_instruction(language_context),
            "Use only the provided web facts and snippets. Do not invent facts.",
            "Keep the answer concise, natural, and in the requested conversation language.",
            "Do not translate URLs. Do not include a Sources section; source links will be appended separately.",
            f"User query:\n{query}",
            f"Draft answer:\n{draft_answer}",
            "Facts:\n" + ("\n".join(fact_lines) if fact_lines else "- No extra structured facts."),
            "Search snippets:\n" + ("\n".join(result_lines) if result_lines else "- No snippets."),
        ]
    )


def _compose_language_matched_web_answer(
    *,
    query: str,
    search_data: Mapping[str, Any],
    query_type: str,
    mode: str,
    language_context: LanguageContext | None,
) -> str:
    draft_answer = build_web_grounded_answer(
        query,
        search_data,
        query_type=query_type,
        language_context=language_context,
    )
    if language_context is None or language_context.code == "en":
        return draft_answer

    draft_body, sources_block = _split_sources_block(draft_answer)
    composer_prompt = _build_grounded_composer_prompt(
        query=query,
        search_data=search_data,
        draft_answer=draft_body or draft_answer,
        language_context=language_context,
    )
    try:
        localized_body = generate_response(
            composer_prompt,
            system_prompt=get_system_prompt(mode, language_context=language_context),
            model=OFFLINE_OLLAMA_MODEL or DEFAULT_MODEL_NAME,
            mode=mode,
            prefer_stream=False,
        )
    except (OllamaError, SonicProviderError, ValueError):
        return draft_answer

    localized_body = sanitize_identity_leaks(localized_body).strip()
    if not localized_body:
        return draft_answer

    if sources_block and "Sources:" not in localized_body:
        return f"{localized_body}\n\n{sources_block}"
    return localized_body


def _run_live_web_provider(
    user_message: str,
    mode: str,
    history: Sequence[Mapping[str, str]] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    runtime_used: str = "online",
    language_context: LanguageContext | None = None,
) -> dict[str, Any]:
    query_type = classify_query(user_message)
    if query_type == QUERY_TYPE_GENERATIVE_TASK:
        return _run_local_provider(
            prompt_text=user_message,
            system_prompt=get_system_prompt(mode, language_context=language_context),
            mode=mode,
            runtime_used=runtime_used,
            user_message=user_message,
            stream_callback=stream_callback,
            language_context=language_context,
        )

    search_data = search_live_web(
        user_message=user_message,
        history=history,
    )
    query = _clean_text(str(search_data.get("query") or user_message))
    reply_text = _compose_language_matched_web_answer(
        query=query,
        search_data=search_data,
        query_type=query_type,
        mode=mode,
        language_context=language_context,
    )
    reply_text = _finalize_reply_text(
        reply_text,
        _build_runtime_identity_reply(runtime_used, language_context=language_context),
    )
    if stream_callback is not None:
        stream_callback(reply_text)

    return _build_provider_response(
        reply_text,
        runtime_used=runtime_used,
        provider_used="serpapi",
        model_used="SONIC Online",
        mode=mode,
        language_context=language_context,
    )


def _build_live_web_no_results_response(
    message: str,
    mode: str,
    runtime_used: str = "online",
    language_context: LanguageContext | None = None,
) -> dict[str, Any]:
    return _build_provider_response(
        message,
        runtime_used=runtime_used,
        provider_used="serpapi",
        model_used="SONIC Online",
        mode=mode,
        language_context=language_context,
    )


def _run_live_web_fallback(
    *,
    prompt_text: str,
    system_prompt: str,
    mode: str,
    user_message: str,
    fallback_reason: str,
    runtime_used: str = "online",
    stream_callback: Callable[[str], None] | None = None,
    language_context: LanguageContext | None = None,
) -> dict[str, Any]:
    if stream_callback is not None:
        stream_callback(f"{ONLINE_FALLBACK_NOTICE}\n\n")

    local_response = _run_local_provider(
        prompt_text=prompt_text,
        system_prompt=system_prompt,
        mode=mode,
        runtime_used=runtime_used,
        user_message=user_message,
        stream_callback=stream_callback,
        language_context=language_context,
    )
    local_response["notice"] = ONLINE_FALLBACK_NOTICE
    local_response["requested_runtime"] = runtime_used
    local_response["fallback_used"] = True
    local_response["fallback_reason"] = fallback_reason or "online_unavailable"
    return local_response


def dispatch_sonic_prompt(
    prompt_text: str,
    system_prompt: str,
    mode: str = DEFAULT_MODE,
    runtime_preference: str = DEFAULT_RUNTIME_PREFERENCE,
    live_web_query: str | None = None,
    live_web_history: Sequence[Mapping[str, str]] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    language_context: LanguageContext | None = None,
) -> dict[str, Any]:
    normalized_mode = _normalize_mode(mode)
    selection = _resolve_runtime_selection(runtime_preference)
    resolved_live_web_query = _clean_text(live_web_query or prompt_text)
    resolved_language_context = language_context or resolve_language_context(
        resolved_live_web_query,
        live_web_history,
    )
    print("DEBUG dispatch runtime_preference =", runtime_preference)
    print("DEBUG dispatch selection =", selection)
    print("DEBUG dispatch mode =", normalized_mode)
    print("DEBUG dispatch live_web_query =", resolved_live_web_query)
    query_type = classify_query(resolved_live_web_query)

    if normalized_mode == "summary":
        return _run_local_provider(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            mode=normalized_mode,
            runtime_used=selection.runtime_used,
            user_message=resolved_live_web_query,
            stream_callback=stream_callback,
            language_context=resolved_language_context,
        )

    if selection.runtime_used == "local":
        return _run_local_provider(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            mode=normalized_mode,
            runtime_used="local",
            user_message=resolved_live_web_query,
            stream_callback=stream_callback,
            language_context=resolved_language_context,
        )

    if query_type == QUERY_TYPE_GENERATIVE_TASK:
     if selection.runtime_used == "online":
        return _run_live_web_provider(
            user_message=resolved_live_web_query,
            mode=normalized_mode,
            history=live_web_history,
            runtime_used="online",
            stream_callback=stream_callback,
            language_context=resolved_language_context,
        )

    return _run_local_provider(
        prompt_text=prompt_text,
        system_prompt=system_prompt,
        mode=normalized_mode,
        runtime_used="local",
        user_message=resolved_live_web_query,
        stream_callback=stream_callback,
        language_context=resolved_language_context,
    )

    try:
        return _run_live_web_provider(
            user_message=resolved_live_web_query,
            mode=normalized_mode,
            history=live_web_history,
            stream_callback=stream_callback,
            runtime_used="online",
            language_context=resolved_language_context,
        )
    except LiveWebConfigError:
        return _run_live_web_fallback(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            mode=normalized_mode,
            user_message=resolved_live_web_query,
            fallback_reason="config_error",
            runtime_used="online",
            stream_callback=stream_callback,
            language_context=resolved_language_context,
        )
    except LiveWebAPIError as exc:
        if exc.category == "no_results":
            if resolved_language_context.code == "en":
                no_results_reply = str(exc)
            else:
                no_results_reply = build_web_grounded_answer(
                    resolved_live_web_query,
                    {"query": resolved_live_web_query, "direct_answer": "", "facts": [], "results": []},
                    query_type=QUERY_TYPE_WEB_LOOKUP,
                    language_context=resolved_language_context,
                )
            return _build_live_web_no_results_response(
                no_results_reply,
                normalized_mode,
                runtime_used="online",
                language_context=resolved_language_context,
            )
        return _run_live_web_fallback(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            mode=normalized_mode,
            user_message=resolved_live_web_query,
            fallback_reason=exc.category or "live_web_unavailable",
            runtime_used="online",
            stream_callback=stream_callback,
            language_context=resolved_language_context,
        )


def ask_sonic(
    message: str,
    mode: str = DEFAULT_MODE,
    history: list[Any] | None = None,
    runtime_preference: str = DEFAULT_RUNTIME_PREFERENCE,
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    print("DEBUG ask_sonic runtime_preference =",runtime_preference)
    prompt_bundle = _build_prompt_bundle(message, mode, history)
    print("DEBUG ask_sonic mode=",prompt_bundle.mode)
    return dispatch_sonic_prompt(
        prompt_text=prompt_bundle.offline_prompt,
        system_prompt=prompt_bundle.system_prompt,
        mode=prompt_bundle.mode,
        runtime_preference=runtime_preference,
        live_web_query=prompt_bundle.live_web_query,
        live_web_history=prompt_bundle.live_web_history,
        stream_callback=stream_callback,
        language_context=prompt_bundle.language_context,
    )

