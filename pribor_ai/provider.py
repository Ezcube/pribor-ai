"""Small, free-model-only OpenRouter client using the Python standard library."""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 45
MAX_ATTEMPTS = 3
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_RETRY_WAIT_SECONDS = 30
MAX_OUTPUT_TOKENS = 8192


class ProviderError(Exception):
    """An actionable public message and a stable code, never a raw API response."""

    def __init__(self, code: str, message: str, *, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Even a same-host redirect is unexpected for the fixed API endpoint.
        # Refusing all redirects prevents leaking Authorization to another host.
        return None


_OPENER = build_opener(_NoRedirect())


def _configuration() -> tuple[str, str]:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()
    if model != "openrouter/free" and not re.fullmatch(
        r"[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+:free", model
    ):
        raise ProviderError(
            "paid_model_blocked",
            "Разрешены только openrouter/free или модели с суффиксом :free. "
            "Платный запрос не отправлен.",
        )
    if not key:
        raise ProviderError(
            "missing_api_key",
            "Задайте OPENROUTER_API_KEY. Ключ нужен и для бесплатных моделей OpenRouter.",
        )
    if any(ord(char) < 33 or ord(char) > 126 for char in key) or len(key) > 2000:
        raise ProviderError("invalid_api_key", "Проверьте формат OPENROUTER_API_KEY.")
    return key, model


def _retry_delay(headers, attempt: int) -> float:
    raw = headers.get("Retry-After") if headers is not None else None
    delay = float(2**attempt)
    if raw is not None:
        try:
            parsed = float(raw)
            if parsed >= 0 and parsed < float("inf"):
                delay = parsed
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(raw)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delay = max(0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    if delay > MAX_RETRY_WAIT_SECONDS:
        raise ProviderError(
            "retry_later",
            "OpenRouter просит подождать более 30 секунд. Повторите запрос позже; "
            "автоматический повтор раньше указанного срока не выполнен.",
        )
    return delay


def _http_error(status: int) -> ProviderError:
    messages = {
        400: ("invalid_request", "OpenRouter отклонил параметры запроса или схему JSON."),
        401: ("authentication_failed", "OpenRouter не принял API-ключ. Проверьте OPENROUTER_API_KEY."),
        402: ("account_limit", "OpenRouter ограничил доступ по состоянию аккаунта. Платная модель не выбрана."),
        403: ("access_denied", "OpenRouter запретил запрос. Проверьте доступ и настройки аккаунта."),
        404: ("model_unavailable", "Бесплатная модель или совместимый провайдер сейчас недоступны."),
        429: ("rate_limited", "Достигнут лимит бесплатных запросов OpenRouter. Повторите позже."),
    }
    if status in messages:
        return ProviderError(*messages[status])
    if 300 <= status < 400:
        return ProviderError("redirect_blocked", "Неожиданное перенаправление OpenRouter заблокировано.")
    if status >= 500:
        return ProviderError("provider_unavailable", "OpenRouter или бесплатная модель временно недоступны.")
    return ProviderError("api_error", f"OpenRouter отклонил запрос (HTTP {status}).")


def _safe_usage(value) -> dict:
    """Copy only numeric token counters/cost, never arbitrary provider fields."""
    if not isinstance(value, dict):
        return {}
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = value.get(key)
        if type(count) is int and 0 <= count <= 10**10:
            usage[key] = count
    cost = value.get("cost")
    if type(cost) in (int, float) and 0 <= cost <= 10**10 and math.isfinite(cost):
        usage["cost"] = cost
    details = value.get("completion_tokens_details")
    if isinstance(details, dict):
        count = details.get("reasoning_tokens")
        if type(count) is int and 0 <= count <= 10**10:
            usage["reasoning_tokens"] = count
    return usage


def _safe_metadata(response: dict) -> dict:
    result = {"usage": _safe_usage(response.get("usage"))}
    model = response.get("model")
    secret = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if (
        isinstance(model, str)
        and len(model) <= 200
        and re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._:/-]+", model)
        and not (secret and secret in model)
    ):
        result["model"] = model
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish = choices[0].get("finish_reason")
        if finish in ("stop", "length", "content_filter", "tool_calls", "error"):
            result["finish_reason"] = finish
    return result


def _parse_response(raw: bytes) -> dict:
    try:
        response = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise ProviderError("invalid_response", "OpenRouter вернул некорректный JSON.") from None
    if not isinstance(response, dict):
        raise ProviderError("invalid_response", "Ответ OpenRouter должен быть JSON-объектом.")
    diagnostics = _safe_metadata(response)

    def error(code: str, message: str) -> ProviderError:
        return ProviderError(code, message, diagnostics=diagnostics)

    if "error" in response:
        # Some providers return an error object after a successful HTTP status.
        raise error("api_error", "OpenRouter сообщил об ошибке генерации. Попробуйте позже.")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise error("empty_response", "OpenRouter не вернул результат генерации.")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise error("truncated_response", "Ответ модели обрезан по лимиту. Карточка не сформирована.")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise error("invalid_response", "Ответ модели имеет неожиданный формат.")
    if message.get("refusal") or choice.get("finish_reason") == "content_filter":
        raise error("model_refusal", "Модель отказалась обрабатывать эти данные.")
    if choice.get("finish_reason") not in ("stop", None):
        raise error("incomplete_response", "Модель не завершила текстовый ответ.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise error("empty_response", "Модель вернула пустой результат.")
    try:
        data = json.loads(content)
    except ValueError:
        raise error("invalid_model_json", "Ответ модели не соответствует формату JSON.") from None
    if not isinstance(data, dict):
        raise error("invalid_model_json", "Ответ модели должен быть JSON-объектом.")
    model = diagnostics.get("model")
    if not model:
        raise error("missing_model", "OpenRouter не указал корректный идентификатор фактически использованной модели.")
    request_id = response.get("id")
    safe_id = request_id if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,200}", request_id) else None
    return {"data": data, "model": model, "request_id": safe_id, "usage": diagnostics["usage"]}


def request_json(system_prompt: str, payload: dict, schema: dict) -> dict:
    """Return {data, model, request_id, usage}; caller validates domain constraints.

    Only the free router or an explicit :free model can be requested. There is
    no paid fallback, no alternate endpoint, and no credential-bearing redirect.
    Retries are limited to transport failures, 429, and 5xx (three total attempts).
    """
    key, model = _configuration()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "product_card_plan", "strict": True, "schema": schema},
        },
        "provider": {
            "require_parameters": True,
            "max_price": {"prompt": 0, "completion": 0, "request": 0},
        },
        # Reasoning and visible JSON share this budget. A 4096-token live
        # response exhausted it; 8192 also fits the smallest current free
        # JSON-capable model's advertised completion limit. Do not globally
        # disable reasoning: some models selected by the free router require it.
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "pribor-ai-local-prototype/1.0",
        },
        method="POST",
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as result:
                raw = result.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ProviderError("response_too_large", "Ответ OpenRouter превысил допустимый размер.")
            return _parse_response(raw)
        except HTTPError as exc:
            status, headers = exc.code, exc.headers
            exc.close()
            if (status == 429 or 500 <= status < 600) and attempt < MAX_ATTEMPTS - 1:
                time.sleep(_retry_delay(headers, attempt))
                continue
            raise _http_error(status) from None
        except (URLError, OSError, HTTPException):
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
                continue
            raise ProviderError(
                "network_error", "Не удалось связаться с OpenRouter после трёх попыток. Проверьте соединение."
            ) from None
    raise AssertionError("Unreachable retry state")
