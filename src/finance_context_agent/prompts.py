"""Prompts for the planner, the answer, and the critic."""

from __future__ import annotations

import json
from typing import Any


def plan_messages(state: dict[str, Any]) -> tuple[str, str]:
    system = (
        "Ты выбираешь подписи строк и периоды для финансового вопроса. "
        "Чисел ячеек у тебя нет, row_key не выдумывай. "
        "Верни JSON с ключами question_type (lookup, compare, explain или compose), "
        "needles (массив подстрок подписи, до трёх, для compose до четырёх — по одной на метрику), "
        "periods (массив объектов period_key, year, phase_year или flag) и "
        "trace (none, precedents или dependents). "
        "precedents — вопрос «почему» или «из чего». dependents — «на что влияет». "
        "Пустой periods, если период не назван. Если человек уже уточнил строку, сузь needles. "
        "Дыра precedent означает trace precedents и question_type explain. "
        "Промах поиска меняет needles, а не выдумывает row_key. "
        "needles — только подписи строк, без слова «какой». Период пиши в periods, не в needles. "
        "explain — только если вопрос про почему или из чего."
    )
    citations = [
        {
            "row_key": item.get("row_key"),
            "label": item.get("label"),
            "period_id": item.get("period_id"),
        }
        for item in state.get("citations") or []
    ]
    user = json.dumps(
        {
            "question": state.get("question"),
            "human_reply": state.get("human_reply") or "",
            "search_note": state.get("search_note") or "",
            "summary": state.get("summary") or {},
            "axes": state.get("axes") or [],
            "prior_citations": citations,
            "gaps": list(state.get("gaps") or []),
        },
        ensure_ascii=False,
    )
    return system, user


def without_account_code(value: Any) -> Any:
    """Drop the parser account code before it reaches a menu, a selection, or the model."""
    if isinstance(value, dict):
        return {
            key: without_account_code(item)
            for key, item in value.items()
            if key != "concept_id"
        }
    if isinstance(value, list):
        return [without_account_code(item) for item in value]
    return value


def answer_messages(state: dict[str, Any]) -> tuple[str, str]:
    observations = without_account_code(list(state.get("observations") or []))
    allowed = {item.get("row_key") for item in observations}
    citations = [item for item in (state.get("citations") or []) if item.get("row_key") in allowed]
    system = (
        "Ответь только по наблюдениям. Каждое число в тексте — это value, "
        "normalized_value или period_id цитаты. Пустую ячейку назови empty или "
        "not_applicable и не подставляй 0. "
        'JSON: {"text": "...", "citations": [{"row_key", "period_id", "cell", '
        '"value", "value_status", "normalized_value"}]}.'
    )
    user = json.dumps(
        {
            "question": state.get("question"),
            "observations": observations,
            "citations": citations,
            "gaps": list(state.get("gaps") or []),
        },
        ensure_ascii=False,
    )
    return system, user


def critic_messages(state: dict[str, Any]) -> tuple[str, str]:
    system = (
        "Найди дыры ответа относительно вопроса. Новых чисел не пиши. "
        "Если текст уже называет value цитаты, верни пустой gaps. "
        "Не проси валюту, масштаб и предшественников. "
        'JSON: {"gaps": ["короткая дыра"]}. Пустой список, если цитаты закрывают вопрос.'
    )
    citations = [
        {
            "row_key": item.get("row_key"),
            "period_id": item.get("period_id"),
            "value": item.get("value"),
            "value_status": item.get("value_status"),
        }
        for item in state.get("proposed_citations") or []
    ]
    user = json.dumps(
        {
            "question": state.get("question"),
            "text": state.get("draft") or "",
            "citations": citations,
            "gaps_already_found": state.get("gaps") or [],
        },
        ensure_ascii=False,
    )
    return system, user
