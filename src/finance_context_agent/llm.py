"""OpenAI-compatible chat client. One JSON retry, then SchemaError."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class QuestionType(StrEnum):
    lookup = "lookup"
    compare = "compare"
    explain = "explain"
    compose = "compose"


class Trace(StrEnum):
    none = "none"
    precedents = "precedents"
    dependents = "dependents"


def _present_period_text(value: str) -> str:
    """A required string still comes back as punctuation when the field is absent."""
    text = value.strip()
    if any(character.isalnum() for character in text):
        return text
    return ""


class Period(BaseModel):
    # A string, not an int: strict output would emit 0, and 0 vetoes the period.
    model_config = ConfigDict(extra="forbid")
    period_key: str = Field(
        default="",
        description="Ключ периода оси. Пустая строка, если поля нет.",
    )
    year: str = Field(default="", description="Календарный год. Пустая строка, если поля нет.")
    phase_year: str = Field(
        default="",
        description="Номер года фазы строкой. Пустая строка, если поля нет.",
    )

    @model_validator(mode="after")
    def blank_placeholders(self) -> Period:
        period_key = _present_period_text(self.period_key)
        year = _present_period_text(self.year)
        phase_year = _present_period_text(self.phase_year)
        if (period_key, year, phase_year) == (self.period_key, self.year, self.phase_year):
            return self
        return self.model_copy(
            update={"period_key": period_key, "year": year, "phase_year": phase_year}
        )


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_type: QuestionType
    needles: list[str] = Field(description="подпись строки как в книге, без текста вопроса")
    periods: list[Period]
    trace: Trace

    @model_validator(mode="after")
    def drop_blank_periods(self) -> Plan:
        filled = [
            period
            for period in self.periods
            if period.period_key.strip() or period.year.strip() or period.phase_year.strip()
        ]
        if len(filled) == len(self.periods):
            return self
        return self.model_copy(update={"periods": filled})


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_key: str
    period_id: str
    cell: str = ""
    value: str = ""
    value_status: str = ""
    normalized_value: str = ""


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    citations: list[Citation]


class Critic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gaps: list[str]


_MODELS: dict[str, type[BaseModel]] = {"plan": Plan, "answer": Answer, "critic": Critic}
_SCHEMA_DROP = {"title", "default", "$defs", "$schema", "$comment"}


def response_format_for(role: str) -> dict[str, Any]:
    """OpenAI Chat Completions structured output for one graph role."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": role,
            "strict": True,
            "schema": _strict_json_schema(_MODELS[role]),
        },
    }


def _strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Flat strict schema. Local grammars reject $ref, and every object property is required."""
    raw = model.model_json_schema()
    schema = _strict_node(raw, raw.get("$defs") or {}, ())
    if not isinstance(schema, dict):
        raise TypeError(schema)
    return schema


def _strict_node(node: Any, defs: dict[str, Any], stack: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_strict_node(item, defs, stack) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        if name in stack:
            raise ValueError(name)
        resolved = _strict_node(defs[name], defs, stack + (name,))
        extras = {key: value for key, value in node.items() if key != "$ref"}
        if extras and isinstance(resolved, dict):
            overlay = _strict_node(extras, defs, stack)
            if isinstance(overlay, dict):
                return {**resolved, **overlay}
        return resolved
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _SCHEMA_DROP:
            continue
        cleaned[key] = _strict_node(value, defs, stack)
    if isinstance(cleaned.get("allOf"), list) and len(cleaned["allOf"]) == 1:
        only = cleaned["allOf"][0]
        if isinstance(only, dict):
            merged = dict(only)
            for key, value in cleaned.items():
                if key != "allOf":
                    merged[key] = value
            cleaned = merged
    properties = cleaned.get("properties")
    if cleaned.get("type") == "object" or isinstance(properties, dict):
        if not isinstance(properties, dict):
            properties = {}
        cleaned["type"] = "object"
        cleaned["properties"] = properties
        cleaned["additionalProperties"] = False
        cleaned["required"] = list(properties)
    return cleaned


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
        model = _MODELS[role]
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
                try:
                    checked = model.model_validate(parsed)
                except ValidationError:
                    last = "invalid"
                    continue
                return checked.model_dump(mode="json")
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
