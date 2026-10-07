"""Prompts for the planner, the book-or-row check, and the answer."""

from __future__ import annotations

import json
from typing import Any


def plan_messages(state: dict[str, Any]) -> tuple[str, str]:
    system = (
        "Ты выбираешь подписи строк и периоды для финансового вопроса. "
        "Чисел ячеек у тебя нет, row_key не выдумывай. "
        "Верни JSON с ключами question_type (lookup, compare, explain или compose), "
        "needles (массив подписей, по одной на метрику, столько, сколько названо в вопросе), "
        "periods (массив объектов period_key, year, phase_year или flag) и "
        "trace (none, precedents или dependents). "
        "precedents — вопрос «почему» или «из чего». dependents — «на что влияет». "
        "Пустой periods, если период не назван. Если человек уже уточнил строку, сузь needles. "
        "Дыра precedent означает trace precedents и question_type explain. "
        "Промах поиска меняет needles, а не выдумывает row_key. "
        "needles — только подписи строк, без слова «какой». Период пиши в periods, не в needles. "
        "explain — если вопрос про почему или из чего, как считается или какая формула."
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


def about_messages(question: str) -> tuple[str, str]:
    """The catalog grounded nothing. book is the whole workbook; row is a missing label."""
    system = (
        "Каталог не содержит подписи из вопроса. "
        "book — вопрос не открывает одну строку и говорит о книге целиком. "
        "row — вопрос требует открыть одну строку, а её подписи в каталоге нет."
    )
    return system, question


def without_account_code(value: Any) -> Any:
    """Drop the parser account code before it reaches a menu, a selection, or the model."""
    if isinstance(value, dict):
        return {
            key: without_account_code(item) for key, item in value.items() if key != "concept_id"
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
        "normalized_value или period_id цитаты, число из formula.text или кэш либо "
        "период прецедента. Пустую ячейку назови empty или not_applicable и не подставляй 0. "
        "Формулу и входы бери только из formula.text и formula.precedents наблюдения. "
        "Если их нет, это сохранённое значение. Не пересчитывай и не подставляй формулу, "
        "которой в наблюдении нет. "
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
