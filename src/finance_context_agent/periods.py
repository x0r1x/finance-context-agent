"""Map a planner period onto one catalog period key."""

from __future__ import annotations

import re
from typing import Any

_FIRST_YEAR = ("первый операцион", "first operational", "phase year 1", "год 1")
_REPAYMENT = ("repayment", "погашен")


def periods_named_in_question(question: str, axes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Period keys written in the question. Used when the planner leaves periods empty."""
    folded_question = question.casefold()
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for axis in axes:
        for period in axis.get("periods") or []:
            key = str(period.get("period_key") or "").strip()
            folded = key.casefold()
            if not folded or folded in seen:
                continue
            match = re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", folded_question)
            if match is None:
                continue
            seen.add(folded)
            found.append((match.start(), key))
    found.sort(key=lambda item: item[0])
    return [{"period_key": key} for _start, key in found]


def resolve_periods(
    requested: list[Any],
    axes: list[dict[str, Any]],
    axis_ids: list[str],
) -> tuple[list[str], str | None]:
    """Return period keys. The error is ``none`` or ``many`` when the axis cannot decide."""
    pool = _pool(axes, axis_ids)
    if not requested:
        return [], None
    chosen: list[str] = []
    for item in requested:
        spec = _clean_spec(_spec(item))
        if not spec:
            return [], "none"
        hits = [period for period in pool if _matches(spec, period)]
        unique = _unique(hits)
        if not unique:
            return [], "none"
        if len(unique) > 1:
            return [], "many"
        chosen.append(unique[0]["period_key"])
    return chosen, None


def _pool(axes: list[dict[str, Any]], axis_ids: list[str]) -> list[dict[str, Any]]:
    allowed = set(axis_ids)
    periods: list[dict[str, Any]] = []
    for axis in axes:
        if allowed and axis.get("id") not in allowed:
            continue
        periods.extend(axis.get("periods") or [])
    return periods


def _clean_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Drop blanks and flags the axis code does not understand."""
    cleaned: dict[str, Any] = {}
    for key in ("period_key", "year", "phase_year", "flag"):
        if key not in spec:
            continue
        value = spec[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key == "flag" and str(value).casefold() != "repayment":
            continue
        cleaned[key] = "repayment" if key == "flag" else value
    return cleaned


def _spec(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    text = str(item).strip()
    folded = text.casefold()
    if any(phrase in folded for phrase in _FIRST_YEAR):
        return {"phase_year": 1}
    if any(phrase in folded for phrase in _REPAYMENT):
        return {"flag": "repayment"}
    if folded.isdigit() and len(folded) == 4:
        return {"year": folded}
    return {"period_key": text}


def _matches(spec: dict[str, Any], period: dict[str, Any]) -> bool:
    if "period_key" in spec and period.get("period_key") != spec["period_key"]:
        return False
    if "year" in spec and not _year_hit(str(spec["year"]), period):
        return False
    if "phase_year" in spec and period.get("phase_year") != _as_int(spec["phase_year"]):
        return False
    if "flag" in spec and not (period.get("flags") or {}).get(spec["flag"]):
        return False
    return any(key in spec for key in ("period_key", "year", "phase_year", "flag"))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _year_hit(year: str, period: dict[str, Any]) -> bool:
    if str(period.get("period_key")) == year:
        return True
    start = period.get("start_date")
    return isinstance(start, str) and start.startswith(year)


def _unique(periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for period in periods:
        key = str(period.get("period_key"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(period)
    return unique
