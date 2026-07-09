from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from config import (
    GENERATED_IMAGES_DIR,
    NVIDIA_API_KEY,
    NVIDIA_IMAGE_API_ENDPOINT,
    NVIDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
    NVIDIA_IMAGE_MAX_RETRIES,
    NVIDIA_IMAGE_MODEL,
    NVIDIA_IMAGE_READ_TIMEOUT_SECONDS,
    NVIDIA_IMAGE_RETRY_BACKOFF_SECONDS,
    NVIDIA_IMAGE_TOTAL_BUDGET_SECONDS,
)


class ImageGenerationError(RuntimeError):
    pass


logger = logging.getLogger(__name__)

RETRYABLE_IMAGE_HTTP_STATUSES = {502, 503, 504}

HTTP_STATUS_MESSAGES = {
    400: "NVIDIA image API rejected the prompt. Please revise the prompt and try again.",
    401: "NVIDIA image API authentication failed. Please check NVIDIA_API_KEY.",
    403: "NVIDIA image API access was denied. Please check your NVIDIA permissions.",
    404: "NVIDIA image generation endpoint was not found. Please verify the image API endpoint and model name.",
    429: "NVIDIA image API rate limited the request. Please wait and try again.",
    500: "NVIDIA image API returned a server error. Please try again.",
    502: "NVIDIA image API is temporarily unavailable. Please try again.",
    503: "NVIDIA image API is temporarily unavailable. Please try again.",
    504: "Image generation is taking too long on the NVIDIA side. Please try again.",
}


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    prompt: str
    model_used: str
    provider_used: str
    image_path: str
    image_url: str = ""
    api_endpoint: str = ""


def _clean_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _preview_text(text: str, limit: int = 120) -> str:
    cleaned_text = " ".join(_clean_text(text).split())
    if len(cleaned_text) <= limit:
        return cleaned_text
    return f"{cleaned_text[: limit - 3].rstrip()}..."


def _slugify(text: str, fallback: str = "image") -> str:
    cleaned_text = _clean_text(text).lower()
    cleaned_text = re.sub(r"[^a-z0-9]+", "-", cleaned_text).strip("-")
    if not cleaned_text:
        return fallback
    return cleaned_text[:48]


def _ensure_output_dir() -> Path:
    GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_IMAGES_DIR


def _guess_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8"):
        return ".jpg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _retry_delay_seconds(attempt_index: int) -> int:
    base_delay = max(0, NVIDIA_IMAGE_RETRY_BACKOFF_SECONDS)
    if base_delay <= 0:
        return 0
    return min(base_delay * attempt_index, 5)


def _remaining_budget_seconds(start_time: float, total_budget: float) -> float:
    return max(0.0, total_budget - (time.monotonic() - start_time))


def _status_error_message(
    status_code: int,
    detail: str = "",
) -> str:
    base_message = HTTP_STATUS_MESSAGES.get(
        status_code,
        f"NVIDIA image API returned HTTP {status_code}.",
    )
    cleaned_detail = (detail or "").strip()
    if cleaned_detail and status_code not in RETRYABLE_IMAGE_HTTP_STATUSES:
        return f"{base_message} {cleaned_detail}"
    return base_message


def _post_image_request(
    payload: bytes,
    api_key: str,
    start_time: float,
    total_budget: float,
    *,
    endpoint: str,
    model: str,
) -> requests.Response:
    for attempt_index in range(1, NVIDIA_IMAGE_MAX_RETRIES + 1):
        remaining = _remaining_budget_seconds(start_time, total_budget)
        logger.debug(
            "Image generation request attempt %s/%s endpoint=%s model=%s remaining_budget=%.2fs payload_bytes=%s",
            attempt_index,
            NVIDIA_IMAGE_MAX_RETRIES,
            endpoint,
            model,
            remaining,
            len(payload),
        )
        if remaining <= 0:
            raise ImageGenerationError(
                "Image generation timed out while waiting for NVIDIA API response. Please try again."
            )

        connect_timeout = float(min(NVIDIA_IMAGE_CONNECT_TIMEOUT_SECONDS, remaining))
        read_timeout = float(min(NVIDIA_IMAGE_READ_TIMEOUT_SECONDS, remaining))

        if connect_timeout < 1.0 or read_timeout < 1.0:
            raise ImageGenerationError(
                "Image generation timed out while waiting for NVIDIA API response. Please try again."
            )

        try:
            response = requests.post(
                endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                timeout=(connect_timeout, read_timeout),
            )
        except requests.exceptions.Timeout as exc:
            logger.debug(
                "Image generation attempt %s timed out after connect_timeout=%.2fs read_timeout=%.2fs",
                attempt_index,
                connect_timeout,
                read_timeout,
            )
            if attempt_index < NVIDIA_IMAGE_MAX_RETRIES:
                sleep_delay = _retry_delay_seconds(attempt_index)
                if _remaining_budget_seconds(start_time, total_budget) <= sleep_delay:
                    raise ImageGenerationError(
                        "Image generation timed out while waiting for NVIDIA API response. Please try again."
                    ) from exc
                time.sleep(sleep_delay)
                continue
            raise ImageGenerationError(
                "Image generation timed out while waiting for NVIDIA API response. Please try again."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            logger.debug(
                "Image generation attempt %s hit a connection error",
                attempt_index,
                exc_info=exc,
            )
            if attempt_index < NVIDIA_IMAGE_MAX_RETRIES:
                sleep_delay = _retry_delay_seconds(attempt_index)
                if _remaining_budget_seconds(start_time, total_budget) <= sleep_delay:
                    raise ImageGenerationError(
                        "Could not connect to the NVIDIA image API. Please check your network and try again."
                    ) from exc
                time.sleep(sleep_delay)
                continue
            raise ImageGenerationError(
                "Could not connect to the NVIDIA image API. Please check your network and try again."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ImageGenerationError(
                f"Could not reach the NVIDIA image API at {endpoint}: {exc}"
            ) from exc

        if response.status_code in RETRYABLE_IMAGE_HTTP_STATUSES and attempt_index < NVIDIA_IMAGE_MAX_RETRIES:
            logger.debug(
                "Image generation attempt %s received retryable HTTP %s",
                attempt_index,
                response.status_code,
            )
            sleep_delay = _retry_delay_seconds(attempt_index)
            response.close()
            if _remaining_budget_seconds(start_time, total_budget) <= sleep_delay:
                if response.status_code == 504:
                    raise ImageGenerationError(
                        "Image generation is taking too long on the NVIDIA side. Please try again."
                    )
                raise ImageGenerationError(_status_error_message(response.status_code))
            time.sleep(sleep_delay)
            continue

        if response.status_code >= 400:
            detail = (response.text or "").strip()
            logger.debug(
                "Image generation attempt %s failed with HTTP %s and detail=%s",
                attempt_index,
                response.status_code,
                _preview_text(detail, limit=160),
            )
            response.close()
            raise ImageGenerationError(_status_error_message(response.status_code, detail))

        logger.debug("Image generation attempt %s succeeded with HTTP %s", attempt_index, response.status_code)
        return response

    raise ImageGenerationError("Image generation failed after retries.")


def _extract_first_entry(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        for key in ("artifacts", "data", "images", "output", "results", "result"):
            candidate = payload.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)) and candidate:
                first_item = candidate[0]
                if isinstance(first_item, Mapping):
                    return first_item
            if isinstance(candidate, Mapping):
                return candidate

        return payload

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)) and payload:
        first_item = payload[0]
        if isinstance(first_item, Mapping):
            return first_item

    return None


def _extract_image_bytes(
    entry: Mapping[str, Any],
    start_time: float,
    total_budget: float,
) -> tuple[bytes, str]:
    for key in ("b64_json", "base64", "image_base64", "content", "blob"):
        candidate = entry.get(key)
        if isinstance(candidate, str) and candidate.strip():
            cleaned_candidate = candidate.strip()
            if cleaned_candidate.startswith("data:") and "," in cleaned_candidate:
                cleaned_candidate = cleaned_candidate.split(",", 1)[1]
            try:
                return base64.b64decode(cleaned_candidate), ""
            except (ValueError, binascii.Error):
                continue

    for key in ("url", "image_url", "href", "download_url"):
        candidate = entry.get(key)
        if isinstance(candidate, str) and candidate.strip():
            image_url = candidate.strip()
            for attempt_index in range(1, NVIDIA_IMAGE_MAX_RETRIES + 1):
                remaining = _remaining_budget_seconds(start_time, total_budget)
                logger.debug(
                    "Image download attempt %s/%s url=%s remaining_budget=%.2fs",
                    attempt_index,
                    NVIDIA_IMAGE_MAX_RETRIES,
                    image_url,
                    remaining,
                )
                if remaining <= 0:
                    raise ImageGenerationError(
                        "Image download timed out while waiting for NVIDIA image data. Please try again."
                    )
                connect_timeout = float(min(NVIDIA_IMAGE_CONNECT_TIMEOUT_SECONDS, remaining))
                read_timeout = float(min(NVIDIA_IMAGE_READ_TIMEOUT_SECONDS, remaining))
                if connect_timeout < 1.0 or read_timeout < 1.0:
                    raise ImageGenerationError(
                        "Image download timed out while waiting for NVIDIA image data. Please try again."
                    )
                try:
                    response = requests.get(
                        image_url,
                        timeout=(connect_timeout, read_timeout),
                    )
                except requests.exceptions.Timeout as exc:
                    logger.debug("Image download attempt %s timed out for url=%s", attempt_index, image_url)
                    if attempt_index < NVIDIA_IMAGE_MAX_RETRIES:
                        sleep_delay = _retry_delay_seconds(attempt_index)
                        if _remaining_budget_seconds(start_time, total_budget) <= sleep_delay:
                            raise ImageGenerationError(
                                "Image download timed out while waiting for NVIDIA image data. Please try again."
                            ) from exc
                        time.sleep(sleep_delay)
                        continue
                    raise ImageGenerationError(
                        "Image download timed out while waiting for NVIDIA image data. Please try again."
                    ) from exc
                except requests.exceptions.ConnectionError as exc:
                    logger.debug(
                        "Image download attempt %s hit a connection error for url=%s",
                        attempt_index,
                        image_url,
                    )
                    if attempt_index < NVIDIA_IMAGE_MAX_RETRIES:
                        sleep_delay = _retry_delay_seconds(attempt_index)
                        if _remaining_budget_seconds(start_time, total_budget) <= sleep_delay:
                            raise ImageGenerationError(
                                "Could not connect to the NVIDIA image download URL. Please check your network and try again."
                            ) from exc
                        time.sleep(sleep_delay)
                        continue
                    raise ImageGenerationError(
                        "Could not connect to the NVIDIA image download URL. Please check your network and try again."
                    ) from exc
                except requests.exceptions.RequestException as exc:
                    raise ImageGenerationError(
                        f"Could not download generated image from NVIDIA: {exc}"
                    ) from exc

                if response.status_code in RETRYABLE_IMAGE_HTTP_STATUSES and attempt_index < NVIDIA_IMAGE_MAX_RETRIES:
                    logger.debug(
                        "Image download attempt %s received retryable HTTP %s for url=%s",
                        attempt_index,
                        response.status_code,
                        image_url,
                    )
                    sleep_delay = _retry_delay_seconds(attempt_index)
                    response.close()
                    if _remaining_budget_seconds(start_time, total_budget) <= sleep_delay:
                        if response.status_code == 504:
                            raise ImageGenerationError(
                                "Image generation is taking too long on the NVIDIA side. Please try again."
                            )
                        raise ImageGenerationError(_status_error_message(response.status_code))
                    time.sleep(sleep_delay)
                    continue

                if response.status_code >= 400:
                    detail = (response.text or "").strip()
                    logger.debug(
                        "Image download attempt %s failed with HTTP %s and detail=%s",
                        attempt_index,
                        response.status_code,
                        _preview_text(detail, limit=160),
                    )
                    response.close()
                    raise ImageGenerationError(_status_error_message(response.status_code, detail))

                content = response.content
                response.close()
                if not content:
                    raise ImageGenerationError("NVIDIA image download returned empty content.")
                logger.debug("Image download succeeded for url=%s (%s bytes)", image_url, len(content))
                return content, image_url

    image_data = entry.get("image")
    if isinstance(image_data, (bytes, bytearray)):
        return bytes(image_data), ""

    raise ImageGenerationError("The image API response did not include image data.")


def _write_image_file(image_bytes: bytes, prompt: str, model: str) -> str:
    output_dir = _ensure_output_dir()
    extension = _guess_extension(image_bytes)
    safe_prompt = _slugify(prompt)
    safe_model = _slugify(model, fallback="model")
    file_name = f"{safe_prompt}-{safe_model}-{uuid.uuid4().hex[:8]}{extension}"
    image_path = output_dir / file_name
    image_path.write_bytes(image_bytes)
    return str(image_path)

def generate_sonic_image(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int | None = None,
) -> ImageGenerationResult:
    start_time = time.monotonic()
    total_budget = float(timeout) if timeout is not None else float(NVIDIA_IMAGE_TOTAL_BUDGET_SECONDS)

    cleaned_prompt = _clean_text(prompt)
    if not cleaned_prompt:
        raise ValueError("Image prompt cannot be empty.")

    api_key = _clean_text(NVIDIA_API_KEY)
    if not api_key:
        raise ImageGenerationError(
            "NVIDIA API key is not configured. Set NVIDIA_API_KEY in .env.local."
        )

    selected_model = _clean_text(model) or NVIDIA_IMAGE_MODEL

    # NVIDIA Flux Schnell endpoint ke liye stable payload
    request_payload: dict[str, Any] = {
    "prompt": cleaned_prompt,
    "width": 1024,
    "height": 1024,
    "seed": 0,
}

    request_data = json.dumps(request_payload).encode("utf-8")

    response = _post_image_request(
        request_data,
        api_key,
        start_time,
        total_budget,
        endpoint=NVIDIA_IMAGE_API_ENDPOINT,
        model=selected_model,
    )

    raw_text = (response.text or "").strip()
    response.close()

    if not raw_text:
        raise ImageGenerationError("NVIDIA image API returned an empty response.")

    try:
        parsed_payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ImageGenerationError(
            f"NVIDIA image API returned invalid JSON: {raw_text[:300]}"
        ) from exc

    entry = _extract_first_entry(parsed_payload)
    if entry is None:
        raise ImageGenerationError(
            f"NVIDIA image API response did not contain image data. Response: {raw_text[:500]}"
        )

    image_bytes, image_url = _extract_image_bytes(entry, start_time, total_budget)
    image_path = _write_image_file(image_bytes, cleaned_prompt, selected_model)

    logger.debug(
        "Image generation completed prompt=%s image_path=%s image_url=%s",
        _preview_text(cleaned_prompt),
        image_path,
        image_url or "-",
    )

    return ImageGenerationResult(
        prompt=cleaned_prompt,
        model_used=selected_model,
        provider_used="nvidia",
        image_path=image_path,
        image_url=image_url,
        api_endpoint=NVIDIA_IMAGE_API_ENDPOINT,
    )