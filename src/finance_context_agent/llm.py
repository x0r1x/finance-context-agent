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

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        self.status = status
        self.body = body[:300]
        super().__init__(message)


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


_SCHEMAS: dict[str, dict[str, Any]] = {
    "plan": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_type": {
                "type": "string",
                "enum": ["lookup", "compare", "explain", "compose"],
            },
            "needles": {
                "type": "array",
                "description": "Только подпись строки, без слова какой. Например EBITDA или Debt.",
                "items": {"type": "string"},
            },
            "periods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "period_key": {"type": "string"},
                        "year": {"type": "string"},
                        "phase_year": {"type": "integer"},
                    },
                },
            },
            "trace": {"type": "string", "enum": ["none", "precedents", "dependents"]},
        },
        "required": ["question_type", "needles", "periods", "trace"],
    },
    "answer": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "row_key": {"type": "string"},
                        "period_id": {"type": "string"},
                        "cell": {"type": "string"},
                        "value": {"type": "string"},
                        "value_status": {"type": "string"},
                        "normalized_value": {"type": "string"},
                    },
                    "required": ["row_key", "period_id"],
                },
            },
        },
        "required": ["text", "citations"],
    },
    "critic": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"gaps": {"type": "array", "items": {"type": "string"}}},
        "required": ["gaps"],
    },
}


def response_format_for(role: str) -> dict[str, Any]:
    """OpenAI Chat Completions structured output for one graph role."""
    return {
        "type": "json_schema",
        "json_schema": {"name": role, "strict": True, "schema": _SCHEMAS[role]},
    }


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
            "response_format": response_format_for(role),
        }
        last = "empty"
        for _ in range(2):
            try:
                response = await self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                raise ModelError(str(exc)) from exc
            if response.status_code >= 400:
                raise ModelError(
                    str(response.status_code),
                    status=response.status_code,
                    body=response.text[:300],
                )
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
