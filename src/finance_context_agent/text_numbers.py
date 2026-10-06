"""Decimal fold and the published scale abbreviation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?%?(?!\w)")
SCALE_WORDS = {
    "k": ("k", "тыс", "thousand"),
    "m": ("m", "млн", "million"),
    "bn": ("bn", "млрд", "billion"),
}


def scale_display(scale: str) -> str:
    words = SCALE_WORDS.get(str(scale or "").casefold())
    if not words or len(words) < 2 or not words[1]:
        return ""
    return f"{words[1]}."


def scale_factor_is_unit(factor: Any) -> bool:
    return factor in (None, 1) or fold_decimal(factor) == "1"


def fold_decimal(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if text.endswith("%"):
        text = text[:-1]
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None
    return format(decimal.normalize(), "f")
