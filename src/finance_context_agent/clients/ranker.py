"""CLM choice client. The body is the TypeSafe system-one request."""

from __future__ import annotations

from typing import Any

import httpx

from finance_context_agent.clients.llm import ModelError
from finance_context_agent.settings import field_default

_INSTRUCTIONS = "Какой один пункт просит реплика?"


class Choice:
    def __init__(self, key: str, probabilities: dict[str, float]) -> None:
        self.key = key
        self.probabilities = probabilities


class RankerClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        seconds = field_default("parser_timeout_sec") if timeout is None else timeout
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(float(seconds)))
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def choose(self, state: str, criteria: dict[str, str]) -> Choice:
        payload = {
            "model": self._model,
            "state": state,
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": _INSTRUCTIONS,
                    "criteria": criteria,
                }
            },
        }
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
        return _choice(response.json(), criteria)


def _choice(body: Any, criteria: dict[str, str]) -> Choice:
    pick = _pick(body)
    key = pick.get("choice")
    raw = pick.get("probabilities")
    if not isinstance(key, str) or key not in criteria or not isinstance(raw, dict):
        raise ModelError("ranker schema")
    probabilities: dict[str, float] = {}
    for name, score in raw.items():
        if name not in criteria or isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ModelError("ranker schema")
        probabilities[str(name)] = float(score)
    if key not in probabilities:
        raise ModelError("ranker schema")
    return Choice(key, probabilities)


def _pick(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ModelError("ranker schema")
    scored = body.get("questions")
    if not isinstance(scored, dict):
        scored = body.get("answers")
    if not isinstance(scored, dict):
        raise ModelError("ranker schema")
    pick = scored.get("pick")
    if not isinstance(pick, dict):
        raise ModelError("ranker schema")
    return pick
