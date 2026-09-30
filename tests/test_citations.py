from finance_context_agent.citations import verify_answer
from tests.fakes import observation


def _check(
    text: str, citations: list[dict], observations: list[dict], **kwargs: object
) -> list[str]:
    return verify_answer(
        question=str(kwargs.get("question") or "Какой DSCR?"),
        question_type=str(kwargs.get("question_type") or "lookup"),
        trace=str(kwargs.get("trace") or "none"),
        text=text,
        citations=citations,
        observations=observations,
    )


def test_matching_number_and_period_pass() -> None:
    obs = observation("row", "2030", "1.50", "C10")
    gaps = _check(
        "В 2030 значение 1.5.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.50"}],
        [obs],
    )
    assert gaps == []


def test_number_outside_citation_and_period_fails() -> None:
    obs = observation("row", "2030", "1.5", "C10")
    gaps = _check(
        "В 2030 значение 1.5, а запас 7.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.5"}],
        [obs],
    )
    assert "number:7" in gaps


def test_period_token_that_is_not_numeric_is_not_a_number() -> None:
    obs = observation("row", "Y5", "1.5", "C10")
    gaps = _check(
        "В Y5 значение 1.5.",
        [{"row_key": "row", "period_id": "Y5", "cell": "C10", "value": "1.5"}],
        [obs],
    )
    assert gaps == []


def test_empty_cache_passes_without_a_number() -> None:
    obs = observation("row", "2030", "", "C10", status="empty")
    gaps = _check(
        "Ячейка empty.",
        [
            {
                "row_key": "row",
                "period_id": "2030",
                "cell": "C10",
                "value": None,
                "value_status": "empty",
            }
        ],
        [obs],
    )
    assert gaps == []


def test_not_applicable_passes_without_zero() -> None:
    obs = observation("row", "2030", "", "C10", status="not_applicable")
    gaps = _check(
        "Ячейка not_applicable.",
        [
            {
                "row_key": "row",
                "period_id": "2030",
                "cell": "C10",
                "value": None,
                "value_status": "not_applicable",
            }
        ],
        [obs],
    )
    assert gaps == []


def test_empty_cited_as_zero_fails() -> None:
    obs = observation("row", "2030", "", "C10", status="empty")
    gaps = _check(
        "Значение 0.",
        [
            {
                "row_key": "row",
                "period_id": "2030",
                "cell": "C10",
                "value": "0",
                "value_status": "empty",
            }
        ],
        [obs],
    )
    assert any(item.startswith("cell:") for item in gaps)
    assert "number:0" in gaps


def test_wrong_cell_fails() -> None:
    obs = observation("row", "2030", "1.5", "C10")
    gaps = _check(
        "1.5",
        [{"row_key": "row", "period_id": "2030", "cell": "Z9", "value": "1.5"}],
        [obs],
    )
    assert "cell:Z9" in gaps


def test_scale_word_required_for_display_value() -> None:
    obs = observation(
        "row", "2030", "1.2", "C10", scale_factor=1_000_000, scale="m", normalized="1200000"
    )
    missing = _check(
        "Значение 1.2.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.2"}],
        [obs],
    )
    assert "scale:row" in missing
    present = _check(
        "Значение 1.2 млн.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.2"}],
        [obs],
    )
    assert present == []


def test_normalized_value_does_not_need_a_scale_word() -> None:
    obs = observation("row", "2030", "1.2", "C10", scale_factor=1000, scale="k", normalized="1200")
    gaps = _check(
        "В шкале книги 1200.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1200"}],
        [obs],
    )
    assert gaps == []


def test_scale_token_is_not_a_substring_of_another_word() -> None:
    obs = observation("row", "2030", "1.2", "C10", scale_factor=1000, scale="m", normalized="1200")
    gaps = _check(
        "The amount is 1.2.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.2"}],
        [obs],
    )
    assert "scale:row" in gaps


def test_compare_needs_two_sides() -> None:
    obs = observation("row", "2030", "1.5", "C10")
    gaps = _check(
        "1.5",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.5"}],
        [obs],
        question_type="compare",
    )
    assert gaps == ["compare_sides"]


def test_why_without_precedent_fails_until_a_precedent_is_cited() -> None:
    bare = observation("row", "2030", "1.5", "C10")
    gaps = _check(
        "Почему 1.5?",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.5"}],
        [bare],
        question="Почему 1.5?",
    )
    assert "precedent" in gaps
    linked = observation("row", "2030", "1.5", "C10", precedents=[{"cell": "A1"}])
    assert (
        _check(
            "Почему 1.5?",
            [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1.5"}],
            [linked],
            question="Почему 1.5?",
        )
        == []
    )


def test_thousands_separator_matches() -> None:
    obs = observation("row", "2030", "1200", "C10")
    gaps = _check(
        "Сумма 1 200.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1200"}],
        [obs],
    )
    assert gaps == []
