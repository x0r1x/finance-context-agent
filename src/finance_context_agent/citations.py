"""Check an answer against the observation slice that produced it."""

from __future__ import annotations

import re
from typing import Any

from finance_context_agent.questions import asks_how
from finance_context_agent.text_numbers import (
    NUMBER,
    SCALE_WORDS,
    fold_decimal,
    scale_factor_is_unit,
)

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
        _allow(allowed, citation.get("value"))
        _allow(allowed, citation.get("normalized_value"))
        _allow_period(allowed, citation.get("period_id"))
        formula_text, precedents = _formula_parts(observation)
        for token in NUMBER.findall(formula_text):
            _allow(allowed, token)
        for precedent in precedents:
            _allow(allowed, precedent.get("value"))
            _allow(allowed, precedent.get("normalized_value"))
            _allow_period(allowed, precedent.get("period_id"))
    for token in NUMBER.findall(text):
        normalized = fold_decimal(token)
        if normalized is not None and normalized in allowed:
            continue
        if token.strip() in allowed:
            continue
        gaps.append(f"number:{token.strip()}")


def scale_words(observation: dict[str, Any]) -> tuple[str, ...]:
    unit = observation.get("unit") or {}
    scale = ""
    if isinstance(unit, dict):
        scale = str(unit.get("scale") or "").casefold()
    if scale in SCALE_WORDS:
        return SCALE_WORDS[scale]
    return (scale,) if scale else ()


def text_has_scale(text: str, words: tuple[str, ...]) -> bool:
    folded = text.casefold()
    for word in words:
        if not word:
            continue
        if re.search(rf"(?<![^\W\d_]){re.escape(word)}(?!\w)", folded):
            return True
        if len(word) >= 3 and re.search(rf"(?<![^\W\d_]){re.escape(word)}", folded):
            return True
    return False


def scale_is_required(citation: dict[str, Any], observation: dict[str, Any]) -> bool:
    if scale_factor_is_unit(observation.get("scale_factor")):
        return False
    if _explicit_zero(citation, observation):
        return False
    if _same(citation.get("value"), observation.get("normalized_value")) and not _same(
        citation.get("value"), observation.get("value")
    ):
        return False
    return bool(scale_words(observation))


def _scale(
    text: str,
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    gaps: list[str],
) -> None:
    for citation, observation in matched:
        if not scale_is_required(citation, observation):
            continue
        words = scale_words(observation)
        if words and not text_has_scale(text, words):
            gaps.append(f"scale:{observation.get('row_key')}")


def _explicit_zero(citation: dict[str, Any], observation: dict[str, Any]) -> bool:
    """A stored zero does not need a scale word. The magnitude is still zero."""
    status = citation.get("value_status") or observation.get("value_status")
    return status == "zero_explicit" and str(citation.get("value")).strip() == "0"


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
    normalized = fold_decimal(raw)
    if normalized is not None:
        allowed.add(normalized)
    if isinstance(raw, str) and raw.strip():
        allowed.add(raw.strip())


def _allow_period(allowed: set[str], raw: Any) -> None:
    """A period id may contain a digit, as in scenario-4. That digit is cited."""
    _allow(allowed, raw)
    if isinstance(raw, str):
        for token in NUMBER.findall(raw):
            _allow(allowed, token)


def _wants_precedent(question: str, question_type: str, trace: str) -> bool:
    del question_type
    if trace == "precedents":
        return True
    return asks_how(question)


def _same(left: Any, right: Any) -> bool:
    if left in (None, "") or right in (None, ""):
        return False
    normalized_left = fold_decimal(left)
    normalized_right = fold_decimal(right)
    if normalized_left is not None and normalized_left == normalized_right:
        return True
    return str(left).strip() == str(right).strip()
