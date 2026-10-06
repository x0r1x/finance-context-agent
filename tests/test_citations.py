import pytest

from finance_context_agent.citations import verify_answer
from finance_context_agent.text_numbers import scale_display
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
    assert scale_display("k") == "тыс."
    assert scale_display("m") == "млн."
    assert scale_display("bn") == "млрд."


def test_explicit_zero_does_not_need_a_scale_word() -> None:
    obs = observation("row", "Y1", "0", "C10", status="zero_explicit", scale_factor=1000, scale="k")
    gaps = _check(
        "EBITDA 0.",
        [{"row_key": "row", "period_id": "Y1", "cell": "C10", "value": "0"}],
        [obs],
        question="Какой EBITDA в Y1?",
    )
    assert "scale:row" not in gaps


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


def test_explain_label_without_a_why_question_does_not_need_a_precedent() -> None:
    obs = observation("row", "Y1", "0", "C10", status="zero_explicit")
    citation = [{"row_key": "row", "period_id": "Y1", "cell": "C10", "value": "0"}]
    gaps = _check(
        "EBITDA 0.",
        citation,
        [obs],
        question="Какой EBITDA в Y1?",
        question_type="explain",
        trace="none",
    )
    assert "precedent" not in gaps


@pytest.mark.parametrize(
    (
        "question",
        "question_type",
        "trace",
        "period",
        "value",
        "status",
        "precedents",
        "formula",
        "draft",
        "gap",
        "exact",
    ),
    [
        pytest.param(
            "Почему EBITDA равен 0?",
            "lookup",
            "none",
            "Y1",
            "0",
            "zero_explicit",
            None,
            None,
            "Почему EBITDA равен 0?",
            False,
            None,
            id="why-bare",
        ),
        pytest.param(
            "Почему EBITDA равен 0?",
            "lookup",
            "none",
            "Y1",
            "0",
            "zero_explicit",
            [{"cell": "A1", "value": "0"}],
            "=A1",
            "Почему EBITDA равен 0?",
            True,
            None,
            id="why-formula-omitted",
        ),
        pytest.param(
            "Почему EBITDA равен 0?",
            "lookup",
            "none",
            "Y1",
            "0",
            "zero_explicit",
            [{"cell": "A1", "value": "0"}],
            "=A1",
            "Почему EBITDA равен 0? Формула C10: =A1. Входы: A1 = 0",
            False,
            None,
            id="why-formula-quoted",
        ),
        pytest.param(
            "Какой EBITDA в Y1?",
            "lookup",
            "precedents",
            "Y1",
            "0",
            "zero_explicit",
            None,
            None,
            "EBITDA 0.",
            False,
            None,
            id="trace-bare",
        ),
        pytest.param(
            "Какой EBITDA в Y1?",
            "lookup",
            "precedents",
            "Y1",
            "0",
            "zero_explicit",
            [{"cell": "A1"}],
            None,
            "EBITDA 0.",
            True,
            None,
            id="trace-linked",
        ),
        pytest.param(
            "Почему 1.5?",
            "lookup",
            "none",
            "2030",
            "1.5",
            "cached",
            None,
            None,
            "Почему 1.5?",
            False,
            None,
            id="why-number-bare",
        ),
        pytest.param(
            "Почему 1.5?",
            "lookup",
            "none",
            "2030",
            "1.5",
            "cached",
            [{"cell": "A1"}],
            None,
            "Почему 1.5?",
            True,
            None,
            id="why-number-linked",
        ),
        pytest.param(
            "Почему 1.5?",
            "lookup",
            "none",
            "2030",
            "1.5",
            "cached",
            [{"cell": "A1"}],
            None,
            "Почему 1.5? Вход A1.",
            False,
            [],
            id="why-number-cited",
        ),
    ],
)
def test_precedent_gap_follows_formula_or_direct_input(
    question: str,
    question_type: str,
    trace: str,
    period: str,
    value: str,
    status: str,
    precedents: list[dict] | None,
    formula: str | None,
    draft: str,
    gap: bool,
    exact: list[str] | None,
) -> None:
    cited = [{"row_key": "row", "period_id": period, "cell": "C10", "value": value}]
    observed = observation("row", period, value, "C10", status=status, precedents=precedents)
    if formula is not None:
        observed["formula"]["text"] = formula
    gaps = _check(
        draft,
        cited,
        [observed],
        question=question,
        question_type=question_type,
        trace=trace,
    )
    assert ("precedent" in gaps) is gap
    if exact is not None:
        assert gaps == exact


def test_thousands_separator_matches() -> None:
    obs = observation("row", "2030", "1200", "C10")
    gaps = _check(
        "Сумма 1 200.",
        [{"row_key": "row", "period_id": "2030", "cell": "C10", "value": "1200"}],
        [obs],
    )
    assert gaps == []
