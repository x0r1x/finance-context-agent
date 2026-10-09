"""Cached cells, laid out as sentences or a table. The model never writes the numbers."""

from __future__ import annotations

from typing import Any

from finance_context_agent.citations import direct_precedents
from finance_context_agent.questions import _pipe_table
from finance_context_agent.text_numbers import scale_display, scale_factor_is_unit

_STATUS_WORDS = {"empty": "пусто", "not_applicable": "не применимо"}
_PERIOD_COLUMNS = 4


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
            return _missing()
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
        if any(not grouped.get(str(row.get("row_key"))) for row in selected):
            return _missing()
        ordered = []
        for row in selected:
            ordered.extend(grouped[str(row.get("row_key"))])
    explain = _explains(state)
    citations = [_line_from_observation(item)[1] for item in ordered]
    draft = render_frame(ordered, "sentence", explain=explain)
    return {
        "draft": draft,
        "citations": citations,
        "missing": False,
        "observations": ordered,
    }


def _missing() -> dict[str, Any]:
    return {
        "draft": "Подтверждённого числа в срезе нет.",
        "citations": [],
        "missing": True,
        "observations": [],
    }


def frame_summary(state: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """What the view model may see. Cell values, addresses, and formulas stay out."""
    labels: list[str] = []
    periods: list[str] = []
    statuses: list[str] = []
    scales: list[str] = []
    for item in observations:
        point, _citation = _point_from_observation(item)
        if point["label"] not in labels:
            labels.append(point["label"])
        if point["period"] not in periods:
            periods.append(point["period"])
        status = str(scalar_observation(item).get("value_status") or "")
        if status not in statuses:
            statuses.append(status)
        if status in _STATUS_WORDS or status == "zero_explicit":
            continue
        word = _published_scale(scalar_observation(item))
        if word and word not in scales:
            scales.append(word)
    return {
        "question": str(state.get("question") or ""),
        "points": len(observations),
        "labels": labels,
        "periods": periods,
        "statuses": statuses,
        "scales": scales,
        "explain": _explains(state),
    }


def legal_views(summary: dict[str, Any], observations: list[dict[str, Any]]) -> list[str]:
    """Orientations this frame can show. A repeated pair would collapse a citation."""
    views = ["sentence"]
    pairs = [_pair(item) for item in observations]
    if len(pairs) != len(set(pairs)):
        return views
    periods = [str(item) for item in (summary.get("periods") or [])]
    points = int(summary.get("points") or 0)
    if points < 2 and not any(periods):
        return views
    views.append("table-period")
    if len(set(periods)) <= _PERIOD_COLUMNS:
        views.append("table-label")
    return views


def render_frame(
    observations: list[dict[str, Any]], view: str, *, explain: bool = False
) -> str:
    points = [_point_from_observation(item)[0] for item in observations]
    if view == "table-period":
        body = _table_period(points)
    elif view == "table-label":
        body = _table_label(points)
    else:
        body = "\n".join(_sentence(point) for point in points)
    if not explain:
        return body
    if view == "sentence" and len(points) == 1:
        return body + "\n" + "\n".join(_formula_lines(observations[0]))
    # A blank line keeps the formula from becoming another table row.
    blocks: list[str] = []
    for item, point in zip(observations, points, strict=True):
        if point["period"]:
            blocks.append(point["period"])
        blocks.extend(_formula_lines(item))
    return body + "\n\n" + "\n".join(blocks)


def _pair(item: dict[str, Any]) -> tuple[str, str]:
    folded = scalar_observation(item)
    return str(folded.get("row_key") or ""), str(folded.get("period_id") or "")


def _table_period(points: list[dict[str, str]]) -> str:
    labels, periods, cells = _grid(points)
    rows = [
        [period, *[cells.get((label, period), "") for label in labels]] for period in periods
    ]
    return _pipe_table(["Период", *labels], rows)


def _table_label(points: list[dict[str, str]]) -> str:
    labels, periods, cells = _grid(points)
    rows = [
        [label, *[cells.get((label, period), "") for period in periods]] for label in labels
    ]
    return _pipe_table(["Подпись", *periods], rows)


def _grid(
    points: list[dict[str, str]],
) -> tuple[list[str], list[str], dict[tuple[str, str], str]]:
    labels: list[str] = []
    periods: list[str] = []
    cells: dict[tuple[str, str], str] = {}
    for point in points:
        if point["label"] not in labels:
            labels.append(point["label"])
        if point["period"] not in periods:
            periods.append(point["period"])
        cells[(point["label"], point["period"])] = point["text"]
    return labels, periods, cells


def _sentence(point: dict[str, str]) -> str:
    if point["period"]:
        return f"{point['label']} в {point['period']}: {point['text']}"
    return f"{point['label']}: {point['text']}"


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
    point, citation = _point_from_observation(item)
    return _sentence(point), citation


def _point_from_observation(item: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
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
    point = {"label": label, "period": period_id, "text": text_value}
    return point, citation


def _published_scale(item: dict[str, Any]) -> str:
    if scale_factor_is_unit(item.get("scale_factor")):
        return ""
    scale = ((item.get("unit") or {}).get("scale") or "")
    return scale_display(scale)
