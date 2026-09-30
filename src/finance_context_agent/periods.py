"""Map a planner period onto one catalog period key."""

from __future__ import annotations

import re
from typing import Any

_FIRST_OPERATION = ("первый операцион", "first operational")
_PHASE_YEAR_ONLY = ("phase year 1", "год 1")
_REPAYMENT_START = ("начала погашен", "начало погашен", "repayment start")
_REPAYMENT = ("repayment", "погашен")
_CALENDAR_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_OPERATION_PHASES = {"operation", "operations", "operating"}


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


def periods_from_question(question: str, axes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keys and phrases written in the question. A blank planner period is not an input."""
    named = periods_named_in_question(question, axes)
    if named:
        return named
    folded = question.casefold()
    if any(phrase in folded for phrase in _REPAYMENT_START):
        return [{"flag": "repayment_start"}]
    if any(phrase in folded for phrase in _FIRST_OPERATION):
        return [{"phase": "operation", "phase_year": "1"}]
    years: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _CALENDAR_YEAR.finditer(question):
        year = match.group(0)
        if year in seen:
            continue
        seen.add(year)
        years.append({"year": year})
    return years


def resolve_periods(
    requested: list[Any],
    axes: list[dict[str, Any]],
    axis_ids: list[str],
) -> tuple[list[str], str | None]:
    """Return period keys. The error is ``none`` or ``many`` when the axis cannot decide."""
    if not requested:
        return [], None
    chosen: list[str] = []
    for item in requested:
        raw = _spec(item)
        spec = _clean_spec(raw)
        if not spec:
            if _unknown_flag_only(raw):
                return [], "none"
            continue
        pool = _pool(axes, axis_ids, spec)
        hits = [period for period in pool if _matches(spec, period)]
        unique = _unique(hits)
        if spec.get("flag") == "repayment_start" and len(unique) > 1:
            unique = unique[:1]
        if not unique:
            return [], "none"
        if len(unique) > 1:
            return [], "many"
        chosen.append(unique[0]["period_key"])
    if not chosen:
        return [], None
    return chosen, None


def _pool(
    axes: list[dict[str, Any]],
    axis_ids: list[str],
    spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not axis_ids:
        return []
    allowed = set(axis_ids)
    own = [axis for axis in axes if axis.get("id") in allowed]
    periods: list[dict[str, Any]] = []
    for axis in own:
        periods.extend(axis.get("periods") or [])
    if spec and _needs_sibling(spec) and not _annotated(periods):
        signature = _signature(own[:1])
        if signature:
            for axis in axes:
                if axis.get("id") in allowed:
                    continue
                sibling = list(axis.get("periods") or [])
                if _signature([axis]) == signature and _annotated(sibling):
                    periods.extend(sibling)
    return periods


def _signature(axes: list[dict[str, Any]]) -> tuple[str, ...]:
    if not axes:
        return ()
    return tuple(
        str(period.get("period_key"))
        for period in (axes[0].get("periods") or [])
        if period.get("period_key")
    )


def _annotated(periods: list[dict[str, Any]]) -> bool:
    for period in periods:
        if period.get("phase"):
            return True
        if period.get("flags"):
            return True
    return False


def _needs_sibling(spec: dict[str, Any]) -> bool:
    return "phase" in spec or "flag" in spec


def _clean_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Drop blanks and flags the axis code does not understand."""
    cleaned: dict[str, Any] = {}
    for key in ("period_key", "year", "phase", "phase_year", "flag"):
        if key not in spec:
            continue
        value = spec[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key == "flag":
            folded = str(value).casefold()
            if folded in ("repayment", "repayment_start"):
                cleaned[key] = folded
            continue
        cleaned[key] = value
    return cleaned


def _unknown_flag_only(raw: dict[str, Any]) -> bool:
    flag = raw.get("flag")
    if flag is None or (isinstance(flag, str) and not str(flag).strip()):
        return False
    if str(flag).casefold() in ("repayment", "repayment_start"):
        return False
    for key in ("period_key", "year", "phase", "phase_year"):
        value = raw.get(key)
        if value is None or (isinstance(value, str) and not str(value).strip()):
            continue
        return False
    return True


def _spec(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    text = str(item).strip()
    folded = text.casefold()
    if any(phrase in folded for phrase in _REPAYMENT_START):
        return {"flag": "repayment_start"}
    if any(phrase in folded for phrase in _FIRST_OPERATION):
        return {"phase": "operation", "phase_year": "1"}
    if any(phrase in folded for phrase in _PHASE_YEAR_ONLY):
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
    if "phase" in spec and not _phase_is(str(spec["phase"]), period.get("phase")):
        return False
    if "phase_year" in spec and period.get("phase_year") != _as_int(spec["phase_year"]):
        return False
    if "flag" in spec and not _flag_hit(str(spec["flag"]), period):
        return False
    return any(key in spec for key in ("period_key", "year", "phase", "phase_year", "flag"))


def _phase_is(wanted: str, actual: Any) -> bool:
    folded = wanted.casefold()
    text = str(actual or "").casefold()
    if folded == "operation":
        return text in _OPERATION_PHASES or text.startswith("operation")
    return text == folded


def _flag_hit(flag: str, period: dict[str, Any]) -> bool:
    flags = period.get("flags") or {}
    if flag == "repayment_start":
        for name, value in flags.items():
            folded = str(name).casefold()
            if value and "repayment" in folded and "start" in folded:
                return True
        return False
    return bool(flags.get(flag))


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
