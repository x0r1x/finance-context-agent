"""Sentence copied from the cached cell, its formula, and its direct inputs."""

from __future__ import annotations

from typing import Any

from finance_context_agent.citations import direct_precedents
from finance_context_agent.text_numbers import scale_display, scale_factor_is_unit

_STATUS_WORDS = {"empty": "пусто", "not_applicable": "не применимо"}


def cache_answer(state: dict[str, Any]) -> dict[str, Any]:
    selected = list(state.get("selected") or [])
    period_ids = [str(item) for item in (state.get("period_ids") or [])]
    wanted = {row.get("row_key") for row in selected}
    scoped = [
        item
        for item in (state.get("observations") or [])
        if item.get("row_key") in wanted
        and (not period_ids or str(item.get("period_id") or "") in period_ids)
    ]
    if period_ids:
        found = {(item.get("row_key"), str(item.get("period_id") or "")) for item in scoped}
        missing = any(
            (row.get("row_key"), period_id) not in found
            for row in selected
            for period_id in period_ids
        )
        if missing:
            return {
                "draft": "Подтверждённого числа в срезе нет.",
                "citations": [],
                "missing": True,
            }
        ordered = []
        for row in selected:
            for period_id in period_ids:
                ordered.append(
                    next(
                        item
                        for item in scoped
                        if item.get("row_key") == row.get("row_key")
                        and str(item.get("period_id") or "") == period_id
                    )
                )
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in scoped:
            grouped.setdefault(str(item.get("row_key")), []).append(item)
        one_each = len(grouped) == len(selected) and all(
            len(items) == 1 for items in grouped.values()
        )
        if not one_each:
            return {
                "draft": "Подтверждённого числа в срезе нет.",
                "citations": [],
                "missing": True,
            }
        ordered = [grouped[str(row.get("row_key"))][0] for row in selected]
    explain = _explains(state)
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for item in ordered:
        line, citation = _line_from_observation(item)
        if explain:
            line = "\n".join([line, *_formula_lines(item)])
        lines.append(line)
        citations.append(citation)
    return {"draft": "\n".join(lines), "citations": citations, "missing": False}


def _explains(state: dict[str, Any]) -> bool:
    plan_body = state.get("plan") or {}
    return plan_body.get("question_type") == "explain" or plan_body.get("trace") == "precedents"


def _formula_lines(item: dict[str, Any]) -> list[str]:
    """Quote the book's formula and its direct inputs. Do not recompute them."""
    formula = item.get("formula") or {}
    if not isinstance(formula, dict):
        formula = {}
    text = str(formula.get("text") or "").strip()
    cell = str((item.get("source") or {}).get("cell") or "").strip()
    directs = direct_precedents(list(formula.get("precedents") or []))
    lines: list[str] = []
    if text:
        lines.append(f"Формула {cell}: {text}" if cell else f"Формула: {text}")
    bits = [bit for bit in (_precedent_bit(entry) for entry in directs) if bit]
    if bits:
        lines.append("Входы: " + "; ".join(bits))
    if not lines:
        lines.append("Формулы в книге нет. Это сохранённое значение.")
    return lines


def _precedent_bit(item: dict[str, Any]) -> str:
    label = str(item.get("label") or "").strip()
    cell = str(item.get("cell") or "").strip()
    status = str(item.get("value_status") or "")
    if status in _STATUS_WORDS:
        shown = _STATUS_WORDS[status]
    elif item.get("value") is None:
        shown = ""
    else:
        shown = str(item.get("value"))
    head = label
    if cell:
        head = f"{head} [{cell}]".strip() if head else cell
    if shown:
        return f"{head} = {shown}" if head else shown
    return head


def scalar_observation(item: dict[str, Any]) -> dict[str, Any]:
    """A params column keyed ``value`` is the scalar slot, not a period on an axis."""
    period = str(item.get("period_id") or "")
    if period.casefold() != "value":
        return item
    return {**item, "period_id": ""}


def _line_from_observation(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    item = scalar_observation(item)
    status = str(item.get("value_status") or "")
    label = str(item.get("label") or item.get("row_key") or "")
    period_id = str(item.get("period_id") or "")
    if status in _STATUS_WORDS:
        text_value = _STATUS_WORDS[status]
        cited_value = ""
    else:
        cited_value = "" if item.get("value") is None else str(item.get("value"))
        text_value = cited_value
        if status != "zero_explicit":
            suffix = _published_scale(item)
            if suffix:
                text_value = f"{text_value} {suffix}"
    if period_id:
        line = f"{label} в {period_id}: {text_value}"
    else:
        line = f"{label}: {text_value}"
    citation: dict[str, Any] = {
        "row_key": item.get("row_key"),
        "period_id": period_id,
        "cell": (item.get("source") or {}).get("cell") or "",
        "value": cited_value,
        "value_status": status,
    }
    normalized = item.get("normalized_value")
    if normalized not in (None, ""):
        citation["normalized_value"] = normalized
    return line, citation


def _published_scale(item: dict[str, Any]) -> str:
    if scale_factor_is_unit(item.get("scale_factor")):
        return ""
    scale = ((item.get("unit") or {}).get("scale") or "")
    return scale_display(scale)
