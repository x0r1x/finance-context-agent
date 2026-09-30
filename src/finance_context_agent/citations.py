"""Check an answer against the observation slice that produced it."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?%?(?!\w)")

_SCALE_WORDS = {
    "k": ("k", "тыс", "thousand"),
    "m": ("m", "млн", "million"),
    "bn": ("bn", "млрд", "billion"),
}

_WHY = ("почему", "из чего", "why")
_INFLUENCE = ("на что влия", "what does it affect")


def verify_answer(
    *,
    question: str,
    question_type: str,
    trace: str,
    text: str,
    citations: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[str]:
    gaps: list[str] = []
    matched = _matched_citations(citations, observations, gaps)
    _numbers(text, matched, gaps)
    _scale(text, matched, gaps)
    if question_type == "compare" and _sides(matched) < 2:
        gaps.append("compare_sides")
    if _wants_precedent(question, question_type, trace) and not _has_precedent(matched):
        gaps.append("precedent")
    return gaps


def wants_influence(question: str, trace: str) -> bool:
    if trace == "dependents":
        return True
    folded = question.casefold()
    return any(phrase in folded for phrase in _INFLUENCE)


def _matched_citations(
    citations: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    gaps: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for citation in citations:
        observation = _find(citation, observations)
        if observation is None or not _citation_ok(citation, observation):
            cell = citation.get("cell") or citation.get("row_key") or "?"
            gaps.append(f"cell:{cell}")
            continue
        matched.append((citation, observation))
    return matched


def _find(citation: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    for observation in observations:
        if observation.get("row_key") != citation.get("row_key"):
            continue
        if observation.get("period_id") != citation.get("period_id"):
            continue
        return observation
    return None


def _citation_ok(citation: dict[str, Any], observation: dict[str, Any]) -> bool:
    source = observation.get("source") or {}
    cell = citation.get("cell")
    if cell and cell != source.get("cell"):
        return False
    status = observation.get("value_status")
    if status in ("empty", "not_applicable"):
        return citation.get("value") in (None, "") and citation.get("value_status") == status
    cited = citation.get("value")
    if cited in (None, ""):
        return False
    return _same(cited, observation.get("value")) or _same(
        cited, observation.get("normalized_value")
    )


def _numbers(
    text: str,
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    gaps: list[str],
) -> None:
    allowed: set[str] = set()
    for citation, _observation in matched:
        for raw in (
            citation.get("value"),
            citation.get("normalized_value"),
            citation.get("period_id"),
        ):
            normalized = _normalize(raw)
            if normalized is not None:
                allowed.add(normalized)
            if isinstance(raw, str):
                allowed.add(raw.strip())
    for token in _NUMBER.findall(text):
        normalized = _normalize(token)
        if normalized is not None and normalized in allowed:
            continue
        if token.strip() in allowed:
            continue
        gaps.append(f"number:{token.strip()}")


def _scale(
    text: str,
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    gaps: list[str],
) -> None:
    for citation, observation in matched:
        factor = observation.get("scale_factor")
        if factor in (None, 1):
            continue
        if _same(citation.get("value"), observation.get("normalized_value")) and not _same(
            citation.get("value"), observation.get("value")
        ):
            continue
        scale = ((observation.get("unit") or {}).get("scale") or "").casefold()
        words = _SCALE_WORDS.get(scale, (scale,) if scale else ())
        if words and not _has_scale(text, words):
            gaps.append(f"scale:{observation.get('row_key')}")


def _has_scale(text: str, words: tuple[str, ...]) -> bool:
    folded = text.casefold()
    for word in words:
        if not word:
            continue
        if re.search(rf"(?<![^\W\d_]){re.escape(word)}(?!\w)", folded):
            return True
        if len(word) >= 3 and re.search(rf"(?<![^\W\d_]){re.escape(word)}", folded):
            return True
    return False


def _sides(matched: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
    identities = {
        (citation.get("row_key"), citation.get("period_id")) for citation, _observation in matched
    }
    return len(identities)


def _has_precedent(matched: list[tuple[dict[str, Any], dict[str, Any]]]) -> bool:
    for _citation, observation in matched:
        precedents = ((observation.get("formula") or {}).get("precedents")) or []
        if precedents:
            return True
    return False


def _wants_precedent(question: str, question_type: str, trace: str) -> bool:
    if question_type == "explain" or trace == "precedents":
        return True
    folded = question.casefold()
    return any(phrase in folded for phrase in _WHY)


def _same(left: Any, right: Any) -> bool:
    if left in (None, "") or right in (None, ""):
        return False
    normalized_left = _normalize(left)
    normalized_right = _normalize(right)
    if normalized_left is not None and normalized_left == normalized_right:
        return True
    return str(left).strip() == str(right).strip()


def _normalize(value: Any) -> str | None:
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
