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


def choice_question(pages: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for needle, page in pages:
        total = int(page.get("total") or 0)
        rows = page.get("rows") or []
        labels = _menu_labels(rows)
        line = f"«{needle}»: {total}. {labels}"
        if total > len(rows):
            line += f" Показаны первые {len(rows)} из {total}."
        lines.append(line)
    return "Какую строку взять?\n" + "\n".join(lines)


def _menu_labels(rows: list[dict[str, Any]]) -> str:
    base = [_label(row) for row in rows]
    counts: dict[str, int] = {}
    for text in base:
        counts[text] = counts.get(text, 0) + 1
    shown: list[str] = []
    for row, text in zip(rows, base, strict=True):
        if counts[text] > 1:
            word = _KIND_WORDS.get(str(row.get("kind") or ""))
            if word:
                text = f"{text}, {word}"
        shown.append(text)
    return "; ".join(shown)


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "")
    sheet = str(row.get("sheet") or "")
    heading = _heading(row.get("label_path"), label)
    mark = ", ".join(part for part in (sheet, heading) if part)
    if not mark:
        return label
    return f"{label} [{mark}]"


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
