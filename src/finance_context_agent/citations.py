"""Check an answer against the observation slice that produced it."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from finance_context_agent.questions import asks_how

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?%?(?!\w)")

_SCALE_WORDS = {
    "k": ("k", "тыс", "thousand"),
    "m": ("m", "млн", "million"),
    "bn": ("bn", "млрд", "billion"),
}

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
    if _wants_precedent(question, question_type, trace) and _explanation_missing(text, matched):
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


def direct_precedents(precedents: list[Any]) -> list[dict[str, Any]]:
    """Inputs of the formula itself. A missing depth is that same first hop."""
    found: list[dict[str, Any]] = []
    for item in precedents:
        if not isinstance(item, dict):
            continue
        depth = item.get("depth")
        if depth in (None, "", 1, "1"):
            found.append(item)
    return found


def _numbers(
    text: str,
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    gaps: list[str],
) -> None:
    allowed: set[str] = set()
    for citation, observation in matched:
        for raw in (
            citation.get("value"),
            citation.get("normalized_value"),
            citation.get("period_id"),
        ):
            _allow(allowed, raw)
        formula_text, precedents = _formula_parts(observation)
        for token in _NUMBER.findall(formula_text):
            _allow(allowed, token)
        for precedent in precedents:
            for raw in (
                precedent.get("value"),
                precedent.get("normalized_value"),
                precedent.get("period_id"),
            ):
                _allow(allowed, raw)
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
        if _explicit_zero(citation, observation):
            continue
        scale = ((observation.get("unit") or {}).get("scale") or "").casefold()
        words = _SCALE_WORDS.get(scale, (scale,) if scale else ())
        if words and not _has_scale(text, words):
            gaps.append(f"scale:{observation.get('row_key')}")


def _explicit_zero(citation: dict[str, Any], observation: dict[str, Any]) -> bool:
    """A stored zero does not need a scale word. The magnitude is still zero."""
    status = citation.get("value_status") or observation.get("value_status")
    return status == "zero_explicit" and str(citation.get("value")).strip() == "0"


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


def _formula_parts(observation: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    formula = observation.get("formula") or {}
    if not isinstance(formula, dict):
        return "", []
    text = str(formula.get("text") or "").strip()
    precedents = [item for item in (formula.get("precedents") or []) if isinstance(item, dict)]
    return text, precedents


def _explanation_missing(text: str, matched: list[tuple[dict[str, Any], dict[str, Any]]]) -> bool:
    folded = text.casefold()
    for _citation, observation in matched:
        formula_text, precedents = _formula_parts(observation)
        if formula_text and formula_text.casefold() not in folded:
            return True
        directs = direct_precedents(precedents)
        cells = [str(item.get("cell") or "").strip() for item in directs]
        cells = [cell for cell in cells if cell]
        if cells and not any(_mentions(folded, cell) for cell in cells):
            return True
        if not cells:
            labels = [str(item.get("label") or "").strip() for item in directs if item.get("label")]
            if labels and not any(_mentions(folded, label) for label in labels):
                return True
    return False


def _mentions(text: str, token: str) -> bool:
    return (
        re.search(rf"(?<![\w]){re.escape(token.casefold())}(?!\w)", text, flags=re.IGNORECASE)
        is not None
    )


def _allow(allowed: set[str], raw: Any) -> None:
    normalized = _normalize(raw)
    if normalized is not None:
        allowed.add(normalized)
    if isinstance(raw, str) and raw.strip():
        allowed.add(raw.strip())


def _wants_precedent(question: str, question_type: str, trace: str) -> bool:
    del question_type
    if trace == "precedents":
        return True
    return asks_how(question)


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
