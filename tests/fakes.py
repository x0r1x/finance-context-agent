"""Stand-ins for the parser and the model. No network."""

from __future__ import annotations

import json
from typing import Any

from finance_context_agent.clients.llm import ModelError, SchemaError
from finance_context_agent.clients.parser import ParserError
from finance_context_agent.clients.ranker import Choice


def catalog_row(
    key: str,
    label: str,
    *,
    sheet: str = "Model",
    concept: str | None = "concept.dscr",
    axis: list[str] | None = None,
    decoy: str = "999",
    label_path: list[str] | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    return {
        "row_key": key,
        "label": label,
        "sheet": sheet,
        "label_path": list(label_path or []),
        "concept_id": concept,
        "axis_ids": axis or ["forecast"],
        "disposition": "mapped",
        "kind": kind,
        "value": decoy,
        "page_marker": "CATALOG_PAGE",
    }


def page(rows: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": "catalog-1",
        "page_marker": "CATALOG_PAGE",
        "total": len(rows) if total is None else total,
        "rows": rows,
    }


def observation(
    row_key: str,
    period_id: str,
    value: str,
    cell: str,
    *,
    status: str = "cached",
    precedents: list[dict[str, Any]] | None = None,
    scale_factor: int | None = None,
    scale: str = "unit",
    normalized: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    return {
        "row_key": row_key,
        "period_id": period_id,
        "label": label or row_key,
        "value": value,
        "normalized_value": value if normalized is None else normalized,
        "value_status": status,
        "scale_factor": scale_factor,
        "unit": {"scale": scale},
        "source": {"cell": cell},
        "formula": {"precedents": precedents or []},
    }


def plan(
    needles: list[str],
    periods: list[Any] | None = None,
    question_type: str = "lookup",
    trace: str = "none",
) -> dict[str, Any]:
    return {
        "question_type": question_type,
        "needles": needles,
        "periods": periods or [],
        "trace": trace,
    }


def answer(text: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"text": text, "citations": citations}


def cite(
    row_key: str,
    period_id: str,
    value: str | None,
    cell: str,
    *,
    status: str = "cached",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "row_key": row_key,
        "period_id": period_id,
        "cell": cell,
        "value_status": status,
    }
    if status in ("empty", "not_applicable"):
        body["value"] = None
    else:
        body["value"] = value
    return body


def year_axes() -> list[dict[str, Any]]:
    return [
        {
            "id": "forecast",
            "periods": [
                {
                    "period_key": "2030",
                    "phase": "operations",
                    "phase_year": 6,
                    "start_date": "2030-01-01",
                },
                {
                    "period_key": "Y1",
                    "phase": "operations",
                    "phase_year": 1,
                    "start_date": "2025-01-01",
                },
                {
                    "period_key": "R1",
                    "phase": "repayment",
                    "phase_year": 2,
                    "flags": {"repayment": True},
                },
            ],
        }
    ]


def two_first_years() -> list[dict[str, Any]]:
    return [
        {
            "id": "forecast",
            "periods": [
                {
                    "period_key": "Y1",
                    "phase": "operations",
                    "phase_year": 1,
                    "start_date": "2025-01-01",
                },
                {
                    "period_key": "Y1b",
                    "phase": "operations",
                    "phase_year": 1,
                    "start_date": "2025-06-01",
                },
            ],
        }
    ]


class FakeParser:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {"marker": "passport", "coverage": {"mapped": 1}}
        self.axes: list[dict[str, Any]] = []
        self.book_etag = "etag-1"
        self.summary_etag = "sum-1"
        self.pages: dict[tuple[str, str], dict[str, Any]] = {}
        self.observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.traces: dict[tuple[str, str], dict[str, Any]] = {}
        self.observation_calls: list[dict[str, Any]] = []
        self.catalog_calls: list[dict[str, Any]] = []
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.list_row_calls: list[str] = []
        self.head_calls: list[str] = []
        self.head_etag: str | None = None
        self.context: dict[str, Any] | None = None
        self.context_calls: list[str] = []
        self.error: ParserError | None = None
        self.force_truncated = False
        self.ready_ok = True

    async def list_jobs(
        self, *, status: str | None = None, q: str | None = None
    ) -> list[dict[str, Any]]:
        self._raise()
        self.list_calls.append({"status": status, "q": q})
        return list(self.jobs)

    async def get_summary(self, job_id: str) -> tuple[dict[str, Any], str]:
        self._raise()
        return self.summary, self.summary_etag

    async def get_context(self, job_id: str) -> dict[str, Any]:
        self._raise()
        self.context_calls.append(job_id)
        if self.context is None:
            raise ParserError(404, "not_found")
        return self.context

    async def head_catalog_etag(self, job_id: str) -> str:
        self._raise()
        self.head_calls.append(job_id)
        return self.head_etag or self.book_etag

    async def get_axes(self, job_id: str) -> tuple[dict[str, Any], str]:
        self._raise()
        return {
            "schema_version": "catalog-1",
            "page_marker": "CATALOG_PAGE",
            "total": 99,
            "axes": self.axes,
            "rows": [{"row_key": "DISCARD_ROW", "label": "discard"}],
        }, self.book_etag

    async def list_rows(self, job_id: str) -> list[dict[str, Any]]:
        self._raise()
        self.list_row_calls.append(job_id)
        if job_id in self.rows:
            return [dict(row) for row in self.rows[job_id]]
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for (found_id, _query), page in self.pages.items():
            if found_id != job_id:
                continue
            for row in page.get("rows") or []:
                key = str(row.get("row_key"))
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(row)
        return ordered

    async def search_rows(self, job_id: str, q: str, *, limit: int = 8) -> dict[str, Any]:
        self._raise()
        self.catalog_calls.append({"job_id": job_id, "q": q, "limit": limit})
        found = self.pages.get((job_id, q.casefold()))
        if found is None:
            return {
                "schema_version": "catalog-1",
                "page_marker": "CATALOG_PAGE",
                "total": 0,
                "rows": [],
            }
        return found

    async def get_observations(
        self,
        job_id: str,
        *,
        row_key: str,
        period_ids: list[str] | None = None,
        precedent_depth: int = 0,
        limit: int = 24,
    ) -> dict[str, Any]:
        self._raise()
        self.observation_calls.append(
            {
                "job_id": job_id,
                "row_key": row_key,
                "period_ids": list(period_ids or []),
                "precedent_depth": precedent_depth,
                "limit": limit,
            }
        )
        rows = [dict(item) for item in self.observations.get((job_id, row_key), [])]
        if period_ids:
            rows = [item for item in rows if item.get("period_id") in period_ids]
        if precedent_depth:
            for item in rows:
                formula = dict(item.get("formula") or {})
                if not formula.get("precedents"):
                    formula["precedents"] = [{"cell": "A1", "row_key": "prec"}]
                item["formula"] = formula
        truncated = self.force_truncated or len(rows) > limit
        return {"observations": rows[:limit], "truncated": truncated}

    async def trace_dependents(self, job_id: str, origin: str, *, depth: int = 2) -> dict[str, Any]:
        self._raise()
        return self.traces.get((job_id, origin), {"nodes": []})

    async def ready(self) -> bool:
        if self.error:
            raise self.error
        return self.ready_ok

    def _raise(self) -> None:
        if self.error is not None:
            raise self.error


class FakeRanker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._queue: list[Any] = []

    def push(
        self,
        key: str,
        probabilities: dict[str, float],
        sheets: float = 0.0,
        hold: float = 0.0,
    ) -> FakeRanker:
        self._queue.append((key, probabilities, sheets, hold))
        return self

    def fail(self) -> FakeRanker:
        self._queue.append("boom")
        return self

    async def choose(
        self,
        state: str,
        criteria: dict[str, str],
        *,
        ask_act: bool = False,
        ask_hold: bool = False,
    ) -> Choice:
        self.calls.append(
            {
                "state": state,
                "criteria": dict(criteria),
                "ask_act": ask_act,
                "ask_hold": ask_hold,
            }
        )
        if not self._queue:
            probs = {name: 0.01 for name in criteria}
            first = next(iter(criteria), "intro")
            return Choice(
                first,
                probs,
                sheets=0.0 if ask_act else None,
                hold=0.0 if ask_hold else None,
            )
        item = self._queue.pop(0)
        if item == "boom":
            raise ModelError("boom")
        key, probabilities, sheets, hold = item
        return Choice(
            key,
            probabilities,
            sheets=sheets if ask_act else None,
            hold=hold if ask_hold else None,
        )


class FakeEmbed:
    """Scripted act scores. An empty queue abstains so a forgotten push reaches chat."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._queue: list[Any] = []

    def push(self, scores: dict[str, float]) -> FakeEmbed:
        self._queue.append(scores)
        return self

    def fail(self) -> FakeEmbed:
        self._queue.append("boom")
        return self

    async def score(self, text: str) -> dict[str, float]:
        self.calls.append(text)
        if not self._queue:
            return {
                "files": 0.01,
                "overview": 0.01,
                "catalog": 0.01,
                "figure": 0.01,
                "greet": 0.01,
                "explain": 0.01,
                "unclear": 0.01,
            }
        item = self._queue.pop(0)
        if item == "boom":
            raise ModelError("embed")
        if not isinstance(item, dict):
            raise AssertionError(item)
        return {str(key): float(value) for key, value in item.items()}


class ScriptedModel:
    def __init__(self) -> None:
        self.queues: dict[str, list[Any]] = {"plan": [], "answer": []}
        self.seen: list[dict[str, str]] = []

    def push(self, role: str, payload: Any) -> ScriptedModel:
        self.queues.setdefault(role, []).append(payload)
        return self

    async def complete_json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        self.seen.append({"role": role, "system": system, "user": user})
        queue = self.queues.setdefault(role, [])
        if not queue:
            raise AssertionError(f"no script for {role}")
        item = queue.pop(0)
        if item == "bad":
            raise SchemaError("bad")
        if item == "boom":
            raise ModelError("boom")
        if not isinstance(item, dict):
            raise AssertionError(item)
        return item


class ByQuestion:
    """Concurrent turns pick a needle from the question text."""

    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []

    async def complete_json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        self.seen.append({"role": role, "system": system, "user": user})
        data = json.loads(user)
        question = str(data.get("question") or "")
        if "ALPHA" in question:
            if role == "plan":
                return plan(["ALPHA"], [{"year": "2030"}])
            return answer("ALPHA 1.11 в 2030.", [cite("row-a", "2030", "1.11", "A1")])
        if role == "plan":
            return plan(["BETA"], [{"year": "2030"}])
        return answer("BETA 2.22 в 2030.", [cite("row-b", "2030", "2.22", "B1")])
