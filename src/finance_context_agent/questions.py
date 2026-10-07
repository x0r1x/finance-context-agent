"""Label text taken from a question. Period rules stay in periods.py."""

from __future__ import annotations

import re
from typing import Any

from finance_context_agent.text_numbers import scale_display

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
# Longer phrases first so a short one does not cut the longer one in half.
_HOW_PHRASES = (
    "по какой формуле",
    "какая формула",
    "как считается",
    "как считаются",
    "как посчитаны",
    "как посчитана",
    "как посчитано",
    "как посчитан",
    "как рассчитаны",
    "как рассчитана",
    "как рассчитано",
    "как рассчитан",
    "из чего",
    "почему",
    "why",
)


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


def asks_how(question: str) -> bool:
    """True when the sentence asks how a row is calculated. The row name stays outside."""
    folded = question.casefold()
    return any(phrase in folded for phrase in _HOW_PHRASES)


def question_needles(question: str, axes: list[dict[str, Any]]) -> list[str]:
    """One needle per label. A conjunction between periods stays one needle."""
    text, _hit = _strip_cover(question)
    text = _strip_how(text)
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


def _strip_how(question: str) -> str:
    text = question
    for phrase in _HOW_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    return text


def is_cover(question: str) -> bool:
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


def cover_question(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    filename = str(summary.get("source_filename") or "").strip()
    if filename:
        lines.append(f"Книга {filename}.")
    lines.append("Какую строку открыть?")
    names = _sheet_names(summary.get("sheets"))
    if len(names) >= 2:
        lines.append("Листы: " + ", ".join(names) + ".")
    formula_count = _positive_count(summary.get("formula_count"))
    if formula_count is not None:
        lines.append(f"Формул: {formula_count}.")
    missing_cache = _positive_count(summary.get("missing_cached_values"))
    if missing_cache is not None:
        lines.append(f"Без сохранённого кэша: {missing_cache}.")
    lines.append("Назовите подписи.")
    return "\n".join(lines)


_STAT_LABELS = (
    ("mapped", "сопоставлено"),
    ("abstract", "заголовки"),
    ("excluded", "исключены"),
    ("abstained", "воздержались"),
)


def book_overview(document: dict[str, Any]) -> str:
    """Copy the stored row summaries. The document itself is not kept."""
    meta = document.get("meta") if isinstance(document.get("meta"), dict) else {}
    workbook = document.get("workbook") if isinstance(document.get("workbook"), dict) else {}
    lines: list[str] = []
    filename = str(meta.get("source_filename") or "").strip()
    if filename:
        lines.append(f"Книга {filename}.")
    names = _sheet_names(workbook.get("sheets"))
    if len(names) >= 2:
        lines.append("Листы: " + ", ".join(names) + ".")
    formula_count = _positive_count(workbook.get("formula_count"))
    if formula_count is not None:
        lines.append(f"Формул: {formula_count}.")
    missing_cache = _positive_count(workbook.get("missing_cached_values"))
    if missing_cache is not None:
        lines.append(f"Без сохранённого кэша: {missing_cache}.")
    lines.extend(_period_lines(document.get("axes")))
    for item in document.get("warnings") or []:
        text = str(item).strip()
        if not text:
            continue
        if not text.endswith("."):
            text += "."
        lines.append(f"Предупреждение: {text}")
    stats = _stats_line(document.get("mapping_stats"))
    if stats:
        lines.append(stats)
    seen_sheet = ""
    for block in document.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        sheet = str(block.get("sheet") or "").strip()
        if sheet and sheet != seen_sheet:
            lines.append(f"Лист {sheet}.")
            seen_sheet = sheet
        for row in block.get("rows") or []:
            line = _overview_row(row)
            if line:
                lines.append(line)
    lines.append("Назовите подписи.")
    return "\n".join(lines)


def _sheet_names(sheets: Any) -> list[str]:
    return [
        str(sheet)
        for sheet in (sheets or [])
        if str(sheet) and not any(char.isdigit() for char in str(sheet))
    ]


def _period_lines(axes: Any) -> list[str]:
    found = [
        axis
        for axis in (axes or [])
        if isinstance(axis, dict) and isinstance(axis.get("periods"), list) and axis["periods"]
    ]
    if len(found) == 1:
        return [f"Периодов: {len(found[0]['periods'])}."]
    lines: list[str] = []
    for axis in found:
        count = len(axis["periods"])
        sheet = str(axis.get("sheet") or "").strip()
        if sheet:
            lines.append(f"Периодов, {sheet}: {count}.")
        else:
            lines.append(f"Периодов: {count}.")
    return lines


def _stats_line(stats: Any) -> str | None:
    if not isinstance(stats, dict):
        return None
    parts = [
        f"{title} {_positive_count(stats.get(key))}"
        for key, title in _STAT_LABELS
        if _positive_count(stats.get(key)) is not None
    ]
    if not parts:
        return None
    return "Строк: " + ", ".join(parts) + "."


def _overview_row(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    label = str(row.get("label") or "").strip()
    if not label:
        return None
    if row.get("kind") == "abstract" or row.get("disposition") == "header":
        return f"{label}."
    numbers = _summary_clause(row.get("numeric_summary"))
    scale = ""
    if numbers:
        hints = row.get("hints") if isinstance(row.get("hints"), dict) else {}
        scale = scale_display(str(hints.get("scale") or ""))
    extras = _side_cells(row)
    head = f"{label}: {numbers}" if numbers else label
    tail = [part for part in (scale, *extras) if part]
    line = head if not tail else f"{head}. " + ". ".join(tail)
    if not line.endswith("."):
        line += "."
    return line


def _summary_clause(summary: Any) -> str:
    if not isinstance(summary, dict):
        return ""
    first = _shown(summary.get("first"))
    last = _shown(summary.get("last"))
    if summary.get("constant") is True or (first is not None and first == last):
        return first or ""
    if first and last:
        text = f"с {first} по {last}"
    elif first:
        text = first
    elif last:
        text = last
    else:
        text = ""
    extra: list[str] = []
    minimum = _shown(summary.get("minimum"))
    maximum = _shown(summary.get("maximum"))
    if minimum is not None and minimum != first:
        extra.append(f"минимум {minimum}")
    if maximum is not None and maximum != last:
        extra.append(f"максимум {maximum}")
    if not extra:
        return text
    if not text:
        return ", ".join(extra)
    return f"{text}, " + ", ".join(extra)


def _side_cells(row: dict[str, Any]) -> list[str]:
    cells = row.get("cells")
    if not isinstance(cells, list):
        return []
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cached = _shown(cell.get("cached_value"))
        if cached is None:
            continue
        if cell.get("role") == "unit":
            parts.append(f"Единица: {cached}")
            continue
        header = str(cell.get("header") or "").strip()
        if header:
            parts.append(f"{header}: {cached}")
    return parts


def _shown(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _positive_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value
