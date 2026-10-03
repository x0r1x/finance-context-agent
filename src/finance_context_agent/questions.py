"""Label text taken from a question. Period rules stay in periods.py."""

from __future__ import annotations

import re
from typing import Any

_MENTION_STOP = frozenset(
    {
        "какой",
        "какая",
        "какое",
        "какие",
        "почему",
        "равен",
        "равна",
        "равно",
        "сравни",
        "в",
        "и",
        "на",
        "за",
        "по",
        "год",
        "года",
        "году",
        "покажи",
        "покажите",
        "первую",
        "первой",
        "первый",
        "первая",
        "первое",
        "операционный",
        "операционного",
        "операционном",
        "начала",
        "начало",
        "погашения",
        "есть",
        "модель",
        "модели",
        "книге",
        "книги",
    }
)
_MENTION_EDGE = "?.!,;:«»\"'"
_COVER_PHRASES = ("ключевые показатели", "общая картина", "обложка")


def question_mention(question: str, axes: list[dict[str, Any]]) -> str:
    """Label text left after period keys, function words, and a bare number."""
    period_keys = {
        str(period.get("period_key") or "").casefold()
        for axis in axes
        for period in axis.get("periods") or []
        if period.get("period_key")
    }
    tokens: list[str] = []
    for raw in question.split():
        token = raw.strip(_MENTION_EDGE)
        folded = token.casefold()
        if len(folded) < 2 or folded in _MENTION_STOP or folded in period_keys:
            continue
        if folded.isdigit():
            continue
        tokens.append(token)
    return " ".join(tokens)


def question_needles(question: str, axes: list[dict[str, Any]]) -> list[str]:
    """One needle per label. A conjunction between periods stays one needle."""
    text, _hit = _strip_cover(question)
    needles: list[str] = []
    for span in _label_spans(text, axes):
        mention = question_mention(span, axes)
        if mention:
            needles.append(mention)
    return needles


def _strip_cover(question: str) -> tuple[str, bool]:
    text = question
    hit = False
    for phrase in _COVER_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(text):
            hit = True
            text = pattern.sub(" ", text)
    return text, hit


def _is_cover(question: str) -> bool:
    folded = question.casefold()
    return any(phrase in folded for phrase in _COVER_PHRASES)


def _label_spans(question: str, axes: list[dict[str, Any]]) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"\s*,\s*", question) if piece.strip()]
    spans: list[str] = []
    for piece in pieces:
        spans.extend(_split_and(piece, axes))
    return spans


def _split_and(piece: str, axes: list[dict[str, Any]]) -> list[str]:
    match = re.search(r"\s+и\s+", piece, flags=re.IGNORECASE)
    if match is None:
        return [piece]
    left, right = piece[: match.start()], piece[match.end() :]
    if question_mention(left, axes) and question_mention(right, axes):
        return _split_and(left, axes) + _split_and(right, axes)
    return [piece]


def _cover_question(summary: dict[str, Any]) -> str:
    names = [
        str(sheet)
        for sheet in (summary.get("sheets") or [])
        if str(sheet) and not any(char.isdigit() for char in str(sheet))
    ]
    lines = ["Какую строку открыть?"]
    if len(names) >= 2:
        lines.append("Листы: " + ", ".join(names) + ".")
    lines.append(
        "Назовите до четырёх: CFADS, обслуживание долга, DSCR, Project IRR, Equity IRR."
    )
    return "\n".join(lines)
