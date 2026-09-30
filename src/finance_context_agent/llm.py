"""OpenAI-compatible chat client. One JSON retry, then SchemaError."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from finance_context_agent.settings import field_default


class SchemaError(Exception):
    """The model answered twice without a JSON object."""


class ModelError(Exception):
    """The model endpoint did not answer."""


class JsonModel(Protocol):
    async def complete_json(self, *, role: str, system: str, user: str) -> dict[str, Any]: ...


def chat_completions_url(base_url: str) -> str:
    parts = urlsplit(base_url.strip())
    path = parts.path.rstrip("/")
    if path == "":
        path = "/v1"
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


class OpenAIChat:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        self._url = chat_completions_url(base_url)
        self._model = model
        self._temperature = field_default("llm_temperature") if temperature is None else temperature
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        seconds = field_default("llm_timeout_sec") if timeout is None else timeout
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(float(seconds)))
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete_json(self, *, role: str, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        last = "empty"
        for _ in range(2):
            try:
                response = await self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                raise ModelError(str(exc)) from exc
            if response.status_code >= 400:
                raise ModelError(f"{response.status_code}")
            content = _message_content(response.json())
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                last = content[:200]
                continue
            if isinstance(parsed, dict):
                return parsed
            last = "not-object"
        raise SchemaError(last)


def _message_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)
