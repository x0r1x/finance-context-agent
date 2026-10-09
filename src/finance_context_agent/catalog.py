"""Keep a catalog row by its own label and phrase the pause menu."""

from __future__ import annotations

import re
from typing import Any

_ACRONYM = re.compile(r"\(([A-Za-z]{2,12})\)\s*$")
_KIND_WORDS = {
    "abstract": "заголовок",
    "fact": "значение",
    "helper": "расчёт",
    "flag": "флаг",
    "params": "параметр",
}


def narrow_hits(mention: str, page: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep a row by its own label. A trailing acronym is the same line."""
    folded = mention.casefold().strip()
    rows = _on_own_label(folded, list(page.get("rows") or []))
    if not folded:
        return rows
    exact = [row for row in rows if str(row.get("label") or "").casefold() == folded]
    if not exact:
        return rows
    longer = [
        row
        for row in rows
        if row not in exact and _keeps_longer(str(row.get("label") or ""), folded)
    ]
    return exact + longer


def _on_own_label(needle: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not needle:
        return rows
    return [row for row in rows if needle in str(row.get("label") or "").casefold()]


def _keeps_longer(label: str, needle: str) -> bool:
    found = _ACRONYM.search(label.strip())
    if found and found.group(1).casefold() == needle:
        return True
    folded = label.casefold()
    if not folded.startswith(needle):
        return False
    return len(folded) == len(needle) or not folded[len(needle)].isalnum()


def narrow_page(mention: str, page: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rows = list(page.get("rows") or [])
    total = int(page.get("total") or 0)
    kept = narrow_hits(mention, page)
    if len(kept) == len(rows):
        return mention, page
    truncated = total > len(rows)
    if truncated and len(kept) <= 1:
        return mention, {**page, "rows": kept, "total": total}
    return mention, {**page, "rows": kept, "total": len(kept)}


MENU_INSTRUCTION = "Какую строку взять? Напишите номер."


def choice_question(pages: list[tuple[str, dict[str, Any]]]) -> str:
    text, _rows = render_menu(pages)
    return text


def render_menu(
    pages: list[tuple[str, dict[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Pause text and the rows in the same order as the numbers."""
    seen: set[str] = set()
    groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    ordered: list[dict[str, Any]] = []
    for needle, page in pages:
        fresh: list[dict[str, Any]] = []
        for row in page.get("rows") or []:
            key = str(row.get("row_key"))
            if key in seen:
                continue
            seen.add(key)
            fresh.append(row)
            ordered.append(row)
        groups.append((needle, page, fresh))
    notes = _kind_notes(ordered)
    index_of = {str(row.get("row_key")): index for index, row in enumerate(ordered, start=1)}
    lines = [MENU_INSTRUCTION]
    for needle, page, fresh in groups:
        total = int(page.get("total") or 0)
        visible = list(page.get("rows") or [])
        header = f"«{needle}»: {total}."
        if total > len(visible):
            header += f" Показаны первые {len(visible)} из {total}."
        lines.append(header)
        for row in fresh:
            lines.extend(_item_lines(index_of[str(row.get("row_key"))], row, notes))
    return "\n".join(lines), ordered


def _heading(path: Any, label: str) -> str:
    if not isinstance(path, list):
        return ""
    folded = label.casefold()
    for part in reversed(path):
        text = str(part).strip()
        if not text or text.casefold() == folded or any(char.isdigit() for char in text):
            continue
        return text
    return ""


def offered_labels(rows: list[dict[str, Any]]) -> str:
    notes = _kind_notes(rows)
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        lines.extend(_item_lines(index, row, notes))
    return "\n".join(lines)


def matching_rows(reply: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose number, № number, label, or printed choice line equals the reply."""
    folded = reply.strip().casefold()
    if not folded:
        return []
    found: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        label = str(row.get("label") or "").strip().casefold()
        choice = _choice_line(index, row).casefold()
        mark = f"№{index}".casefold()
        if folded in {str(index), mark, choice} or (label and folded == label):
            found.append(row)
    return found


def menu_criterion(index: int, row: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """One shown menu line plus its sheet and heading, for a closed choice."""
    notes = _kind_notes(rows)
    parts = [_choice_line(index, row)]
    detail = _detail(row, notes.get(str(row.get("row_key")), ""))
    if detail:
        parts.append(detail)
    return " ".join(parts)


def menu_choices(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Shown items for a closed choice. No row key and no cell value."""
    notes = _kind_notes(rows)
    choices: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        label = str(row.get("label") or "").strip()
        item = {
            "number": str(index),
            "label": label,
            "sheet": str(row.get("sheet") or "").strip(),
            "heading": _heading(row.get("label_path"), label),
        }
        word = notes.get(str(row.get("row_key")), "")
        if word:
            item["kind"] = word
        choices.append(item)
    return choices


def _item_lines(index: int, row: dict[str, Any], notes: dict[str, str]) -> list[str]:
    lines = [_choice_line(index, row)]
    detail = _detail(row, notes.get(str(row.get("row_key")), ""))
    if detail:
        lines.append(detail)
    return lines


def _choice_line(index: int, row: dict[str, Any]) -> str:
    return f"№{index} {str(row.get('label') or '').strip()}"


def _detail(row: dict[str, Any], kind_word: str) -> str:
    sheet = str(row.get("sheet") or "").strip()
    heading = _heading(row.get("label_path"), str(row.get("label") or ""))
    sentences: list[str] = []
    if sheet:
        sentences.append(f"Лист {sheet}.")
    if heading:
        sentences.append(f"Раздел {heading}.")
    if kind_word:
        sentences.append(f"Это {kind_word}.")
    return " ".join(sentences)


def _kind_notes(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts: dict[tuple[str, str, str], int] = {}
    identities = [_identity(row) for row in rows]
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    notes: dict[str, str] = {}
    for row, identity in zip(rows, identities, strict=True):
        if counts[identity] < 2:
            continue
        word = _KIND_WORDS.get(str(row.get("kind") or ""))
        if word:
            notes[str(row.get("row_key"))] = word
    return notes


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    label = str(row.get("label") or "").strip().casefold()
    sheet = str(row.get("sheet") or "").strip().casefold()
    heading = _heading(row.get("label_path"), str(row.get("label") or "")).casefold()
    return label, sheet, heading


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row.get("row_key"),
        "label": row.get("label"),
        "sheet": row.get("sheet"),
        "label_path": list(row.get("label_path") or []),
        "axis_ids": list(row.get("axis_ids") or []),
        "disposition": row.get("disposition"),
        "kind": row.get("kind"),
    }
