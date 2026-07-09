from __future__ import annotations

from core.language import LanguageContext, build_language_instruction


PROMPT_KIND_HINTS = {
    "anime": ("anime:", "anime style:", "manga:"),
    "cinematic": ("cinematic:", "movie:", "film:", "scene:"),
    "creative_writing": ("creative writing:", "writing:", "writing prompt:", "narrative:"),
    "image": ("image:", "art:", "picture:", "visual:", "illustration:"),
}


def _clean_text(text: str) -> str:
    return (text or "").strip()


def _detect_prompt_kind(idea: str, explicit_kind: str | None = None) -> tuple[str, str]:
    cleaned_idea = _clean_text(idea)
    if not cleaned_idea:
        raise ValueError("Prompt idea cannot be empty.")

    if explicit_kind and explicit_kind.strip().lower() in PROMPT_KIND_HINTS:
        return explicit_kind.strip().lower(), cleaned_idea

    lowered_idea = cleaned_idea.lower()
    for kind_name, prefixes in PROMPT_KIND_HINTS.items():
        for prefix in prefixes:
            if lowered_idea.startswith(prefix):
                remainder = cleaned_idea[len(prefix) :].strip()
                return kind_name, remainder or cleaned_idea

    if "anime" in lowered_idea:
        return "anime", cleaned_idea
    if "cinematic" in lowered_idea or "movie" in lowered_idea or "film" in lowered_idea:
        return "cinematic", cleaned_idea
    if "write" in lowered_idea or "writing" in lowered_idea:
        return "creative_writing", cleaned_idea

    return "image", cleaned_idea


def build_prompt_generation_prompt(
    idea: str,
    prompt_kind: str | None = None,
    language_context: LanguageContext | None = None,
) -> str:
    kind_name, cleaned_idea = _detect_prompt_kind(idea, explicit_kind=prompt_kind)

    if kind_name == "anime":
        kind_rules = "Create an anime-style prompt with clear character, scene, mood, and visual detail."
    elif kind_name == "cinematic":
        kind_rules = "Create a cinematic prompt with camera feel, lighting, mood, and composition."
    elif kind_name == "creative_writing":
        kind_rules = "Create a creative writing prompt with a clear premise, tone, and inspiration cues."
    else:
        kind_rules = "Create a polished image prompt with subject, environment, style, lighting, mood, and composition."

    language_rules = ""
    if language_context is not None:
        language_rules = f"{build_language_instruction(language_context)}\n"

    return (
        "Create a prompt from the user's idea.\n"
        f"{language_rules}"
        f"Prompt kind: {kind_name}.\n"
        f"{kind_rules}\n"
        "Make it specific, vivid, and production-ready.\n"
        "Return only the finished prompt.\n\n"
        f"Idea:\n{cleaned_idea}"
    )
