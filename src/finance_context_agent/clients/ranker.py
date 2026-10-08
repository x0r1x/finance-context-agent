"""CLM choice client. The body is the TypeSafe system-one request."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from finance_context_agent.clients.llm import ModelError
from finance_context_agent.questions import LIST_FLOOR
from finance_context_agent.settings import field_default

logger = logging.getLogger(__name__)

_INSTRUCTIONS = "Какой один пункт просит реплика?"
_ACT_INSTRUCTIONS = "Что просит реплика?"
_ACT_VALUES = "Одно число или ряд по периодам."
_ACT_SHEETS = "Какие есть строки и на каких они листах."


class Choice:
    def __init__(
        self,
        key: str,
        probabilities: dict[str, float],
        sheets: float | None = None,
    ) -> None:
        self.key = key
        self.probabilities = probabilities
        self.sheets = sheets


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

    async def choose(
        self, state: str, criteria: dict[str, str], *, ask_act: bool = False
    ) -> Choice:
        questions: dict[str, Any] = {
            "pick": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": criteria,
            }
        }
        if ask_act:
            questions["act"] = {
                "type": "choice",
                "instructions": _ACT_INSTRUCTIONS,
                "criteria": {"values": _ACT_VALUES, "sheets": _ACT_SHEETS},
            }
        payload = {"model": self._model, "state": state, "questions": questions}
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
        body = response.json()
        choice = _choice(body, criteria)
        if ask_act:
            choice.sheets = _sheets(body)
        logged = choice.key
        if ask_act and choice.sheets is not None and choice.sheets >= LIST_FLOOR:
            logged = "sheets"
        logger.info("ranker post %s %s", self._url, logged)
        return choice


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


def _scored(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ModelError("ranker schema")
    scored = body.get("questions")
    if not isinstance(scored, dict):
        scored = body.get("answers")
    if not isinstance(scored, dict):
        raise ModelError("ranker schema")
    return scored


def _pick(body: Any) -> dict[str, Any]:
    pick = _scored(body).get("pick")
    if not isinstance(pick, dict):
        raise ModelError("ranker schema")
    return pick


def _sheets(body: Any) -> float:
    act = _scored(body).get("act")
    if not isinstance(act, dict):
        raise ModelError("ranker schema")
    raw = act.get("probabilities")
    if not isinstance(raw, dict):
        raise ModelError("ranker schema")
    score = raw.get("sheets")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ModelError("ranker schema")
    return float(score)
