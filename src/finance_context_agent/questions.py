"""Label text taken from a question. Period rules stay in periods.py."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from finance_context_agent.catalog import _keeps_longer
from finance_context_agent.text_numbers import display_cached, fold_decimal, scale_display

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


PICK_FLOOR = 0.55
PICK_GAP = 0.15
MENU_FLOOR = 0.25
SHORTLIST_CAP = 24
MENU_CAP = 8

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _bounded(haystack: str, needle: str) -> bool:
    """True when needle sits in haystack with a non-letter on each side."""
    if not needle:
        return False
    start = 0
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return False
        end = found + len(needle)
        before = found == 0 or not haystack[found - 1].isalnum()
        after = end == len(haystack) or not haystack[end].isalnum()
        if before and after:
            return True
        start = found + 1


def contained_labels(text: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose label is a bounded piece of the reply.

    A shorter label is dropped when a longer matched label contains it.
    """
    folded = text.casefold()
    matched: list[dict[str, Any]] = []
    folded_labels: list[str] = []
    for row in rows:
        label = str(row.get("label") or "").strip().casefold()
        if not label or not _bounded(folded, label):
            continue
        matched.append(row)
        folded_labels.append(label)
    dropped: set[int] = set()
    for index, label in enumerate(folded_labels):
        for other_index, other in enumerate(folded_labels):
            if other_index == index or len(other) <= len(label):
                continue
            if _bounded(other, label):
                dropped.add(index)
                break
    return [row for index, row in enumerate(matched) if index not in dropped]


def same_line(hits: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A single contained label also offers the longer line that continues it.

    Two different labels in one reply stay as they are. A longer label is the
    same line when it starts with the short one or ends with it in parentheses.
    """
    labels: list[str] = []
    for row in hits:
        label = str(row.get("label") or "").strip().casefold()
        if label and label not in labels:
            labels.append(label)
    if len(labels) != 1:
        return hits
    needle = labels[0]
    widened: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        other = str(row.get("label") or "").strip()
        key = str(row.get("row_key") or "")
        if not other or not key or key in seen:
            continue
        if other.casefold() == needle or _keeps_longer(other, needle):
            seen.add(key)
            widened.append(row)
    return widened or hits


def shortlist_rows(
    text: str, rows: list[dict[str, Any]], cap: int = SHORTLIST_CAP
) -> list[dict[str, Any]]:
    """Rows whose label contains a word of the reply. Highest overlap first."""
    words: list[str] = []
    seen: set[str] = set()
    for word in _WORD.findall(text.casefold()):
        if len(word) < 3 or word in seen:
            continue
        seen.add(word)
        words.append(word)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        label = str(row.get("label") or "").casefold()
        hits = sum(1 for word in words if _bounded(label, word))
        if hits:
            scored.append((hits, -index, row))
    scored.sort(reverse=True)
    return [row for _hits, _index, row in scored[:cap]]


def decide_margin(
    probabilities: dict[str, float],
    keys: set[str],
    *,
    menu_open: bool,
) -> dict[str, Any]:
    """Take a winner, keep an open menu, or offer the rows that cleared the floor."""
    if not probabilities:
        return {"act": "again"} if menu_open else {"act": "miss"}
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    winner, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    if best >= PICK_FLOOR and best - second >= PICK_GAP:
        return {"act": "take", "key": str(winner)}
    if menu_open:
        return {"act": "again"}
    ranked = [str(key) for key, score in ordered if key in keys and score >= MENU_FLOOR]
    if ranked:
        return {"act": "menu", "keys": ranked[:MENU_CAP]}
    return {"act": "miss"}


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


_FILE_EXTENSIONS = frozenset({"xlsx", "xlsm"})
_NAME_SPLIT = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")


def name_tokens(text: str) -> list[str]:
    """Pieces of a filename or a reply, split on everything that is not a letter or digit."""
    return [part for part in _NAME_SPLIT.split(text.casefold()) if part]


def file_segments(name: str) -> list[str]:
    """Stem pieces of a workbook name. The extension is not a name."""
    return [part for part in name_tokens(name) if part not in _FILE_EXTENSIONS]


def remainder_needles(
    question: str, names: list[str], axes: list[dict[str, Any]]
) -> list[str]:
    """Needles left after the named workbook is removed."""
    text = question
    segments: list[str] = []
    for name in names:
        cleaned = name.strip()
        if not cleaned:
            continue
        text = re.sub(re.escape(cleaned), " ", text, flags=re.IGNORECASE)
        segments.extend(file_segments(cleaned))
    for segment in sorted(set(segments), key=len, reverse=True):
        text = re.sub(
            rf"(?<![0-9A-Za-zА-Яа-яЁё]){re.escape(segment)}(?![0-9A-Za-zА-Яа-яЁё])",
            " ",
            text,
            flags=re.IGNORECASE,
        )
    return question_needles(text, axes)


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


def books_reply(jobs: list[dict[str, Any]]) -> str:
    """Finished filenames only. No sheet list and no cell figures."""
    lines = ["Готовые книги:"]
    for job in jobs:
        name = str(job.get("source_filename") or job.get("job_id") or "").strip()
        if name:
            lines.append(f"- {name}")
    return "\n".join(lines)


def chitchat_reply() -> str:
    """Who the agent is. No book name, no sheet list, and no cell figures."""
    return "\n".join(
        [
            "Я finance-context-agent.",
            "Читаю сохранённые числа книги и отвечаю только по ним.",
            "Могу найти строку по подписи, показать её по периодам, "
            "сузить до названного периода и объяснить формулу из кэша.",
            "Обзор книги даю, когда о нём просят.",
            "Напишите, что посмотреть.",
        ]
    )


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


_DATE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
_LIVE = frozenset({"mapped", "abstained"})


def book_overview(document: dict[str, Any]) -> str:
    """Sections and stored figures of the live rows. The document is not kept."""
    meta = document.get("meta") if isinstance(document.get("meta"), dict) else {}
    lines: list[str] = []
    filename = str(meta.get("source_filename") or "").strip()
    if filename:
        lines.append(filename)
    pending: str | None = None
    last_heading = ""
    for block in document.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        for row in block.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if _is_heading(row):
                label = _label(row)
                if label:
                    pending = label
                continue
            rendered = _metric_line(row)
            if rendered is None:
                continue
            if pending and pending != last_heading:
                lines.append(f"## {pending}")
                last_heading = pending
            pending = None
            lines.append(rendered)
    return "\n".join(lines)


def _sheet_names(sheets: Any) -> list[str]:
    return [
        str(sheet)
        for sheet in (sheets or [])
        if str(sheet) and not any(char.isdigit() for char in str(sheet))
    ]


def _is_heading(row: dict[str, Any]) -> bool:
    if row.get("hidden") is True or row.get("disposition") == "excluded":
        return False
    return row.get("kind") == "abstract" or row.get("disposition") == "header"


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "").strip()
    if not label or label.casefold() == "none":
        return ""
    return label


def _metric_line(row: dict[str, Any]) -> str | None:
    if row.get("disposition") not in _LIVE or row.get("hidden") is True:
        return None
    if row.get("article_role") == "check":
        return None
    label = _label(row)
    if not label:
        return None
    unit = _first_unit(row)
    percent = _scale_percent(row.get("numeric_summary"), bool(unit and "%" in unit))
    numbers, endpoints = _summary_text(row.get("numeric_summary"), percent=percent)
    if unit.casefold() == "date":
        numbers = ""
        endpoints = set()
        unit = ""
    if percent and unit:
        rest = unit.replace("%", "").strip()
        if rest and numbers:
            numbers = f"{numbers} {rest}"
        unit = ""
    elif unit:
        unit = unit.replace("'000", "").replace("’000", "").strip()
    dates, notes, loose = _side_notes(row, percent=percent, endpoints=endpoints)
    if _blank_figure(numbers):
        numbers = ""
    if not numbers and loose:
        numbers = loose[0]
        notes = loose[1:] + notes
    scale = ""
    if numbers or any(char.isdigit() for note in notes for char in note):
        hints = row.get("hints") if isinstance(row.get("hints"), dict) else {}
        scale = scale_display(str(hints.get("scale") or ""))
    measure = " ".join(part for part in (scale, unit) if part)
    span = _date_span(dates)
    if numbers:
        parts = [numbers, measure, span, *notes]
    else:
        parts = [*notes, measure, span]
    if not any(parts):
        return None
    body = ", ".join([part for part in parts if part])
    if not body:
        return None
    return f"- {label} — {body}"


def _scale_percent(summary: Any, percent: bool) -> bool:
    """Scale the row only when every stored figure is a fraction."""
    if not percent:
        return False
    if not isinstance(summary, dict):
        return True
    for key in ("first", "last", "minimum", "maximum"):
        folded = fold_decimal(summary.get(key))
        if folded is None:
            continue
        if abs(Decimal(folded)) > 1:
            return False
    return True


def _zero_end(text: str | None) -> bool:
    return text in (None, "0", "0%")


def _blank_figure(numbers: str) -> bool:
    head, _, _rest = numbers.partition(" ")
    return head in ("", "0", "0%")


def _summary_text(summary: Any, *, percent: bool) -> tuple[str, set[str]]:
    if not isinstance(summary, dict):
        return "", set()
    shown = {
        key: display_cached(summary.get(key), percent=percent)
        for key in ("first", "last", "minimum", "maximum")
    }
    ends = {text for text in shown.values() if text}
    first, last = shown["first"], shown["last"]
    minimum, maximum = shown["minimum"], shown["maximum"]
    if summary.get("constant") is True or (ends and len(ends) == 1):
        return first or last or minimum or maximum or "", ends
    if (first is not None or last is not None) and _zero_end(first) and _zero_end(last):
        if _zero_end(minimum) and _zero_end(maximum):
            return "0", ends
        if not _zero_end(maximum):
            text = f"до {maximum}"
            if not _zero_end(minimum) and minimum != maximum:
                text += f", минимум {minimum}"
            return text, ends
        if not _zero_end(minimum):
            return f"минимум {minimum}", ends
        return "0", ends
    if first and last:
        text = f"с {first} по {last}"
    elif first:
        text = first
    elif last:
        text = last
    else:
        text = ""
    extra: list[str] = []
    if minimum not in (None, first, last):
        extra.append(f"минимум {minimum}")
    if maximum not in (None, first, last):
        extra.append(f"максимум {maximum}")
    if extra and text:
        text = f"{text}, " + ", ".join(extra)
    elif extra:
        text = ", ".join(extra)
    return text, ends


def _first_unit(row: dict[str, Any]) -> str:
    for cell in _cells(row):
        if cell.get("role") != "unit":
            continue
        text = str(cell.get("cached_value") or "").strip()
        if text:
            return text
    return ""


def _side_notes(
    row: dict[str, Any], *, percent: bool, endpoints: set[str]
) -> tuple[list[str], list[str], list[str]]:
    dates: list[str] = []
    notes: list[str] = []
    loose: list[str] = []
    for cell in _cells(row):
        if cell.get("role") == "unit":
            continue
        cached = str(cell.get("cached_value") or "").strip()
        if not cached or cached.casefold() == "none":
            continue
        if _DATE.match(cached):
            dates.append(cached)
            continue
        plain = display_cached(cached, percent=False)
        as_series = display_cached(cached, percent=percent)
        if plain in ("0", "1") or as_series in ("0", "1", "0%", "100%"):
            continue
        if (as_series is not None and as_series in endpoints) or (
            plain is not None and plain in endpoints
        ):
            continue
        header = str(cell.get("header") or "").strip()
        text = as_series if percent and as_series is not None else plain
        if text is None:
            text = cached
        if plain is not None and not header:
            if endpoints:
                continue
            loose.append(text)
            continue
        notes.append(f"{header}: {text}" if header else text)
    return dates, notes, loose


def _date_span(dates: list[str]) -> str:
    if len(dates) >= 2:
        return f"{dates[0]}–{dates[1]}"
    if dates:
        return dates[0]
    return ""


def _cells(row: dict[str, Any]) -> list[dict[str, Any]]:
    cells = row.get("cells")
    if not isinstance(cells, list):
        return []
    return [cell for cell in cells if isinstance(cell, dict)]


def _positive_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value
