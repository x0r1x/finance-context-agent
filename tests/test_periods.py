import pytest

from finance_context_agent.periods import (
    periods_from_question,
    periods_named_in_question,
    resolve_periods,
)
from tests.fakes import two_first_years, year_axes

_PL = [{"id": "P&L!r2", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}]
_Y1_Y10 = [{"id": "forecast", "periods": [{"period_key": "Y1"}, {"period_key": "Y10"}]}]
_Y1_Y2 = [{"id": "forecast", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}]
_Y5_Y10 = [{"id": "forecast", "periods": [{"period_key": "Y5"}, {"period_key": "Y10"}]}]
_ONLY_Y1 = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
_CALENDAR_KEY = [{"id": "forecast", "periods": [{"period_key": "2030"}]}]
_TIMELINE = [
    {
        "id": "timeline",
        "periods": [
            {"period_key": "2024", "phase": "construction", "phase_year": 1},
            {"period_key": "2026", "phase": "operation", "phase_year": 1},
        ],
    }
]
_BARE = [{"period_key": "Y1"}, {"period_key": "Y5"}]
_TBA = [
    {
        "period_key": "Y1",
        "phase": "construction",
        "phase_year": 1,
        "flags": {"repayment start date": False},
    },
    {
        "period_key": "Y5",
        "phase": "operation",
        "phase_year": 1,
        "flags": {"repayment start date": True, "repayment": True},
    },
]
_SIBLING = [{"id": "P&L!r2", "periods": _BARE}, {"id": "TBA!r2", "periods": _TBA}]


@pytest.mark.parametrize(
    ("wanted", "axes", "axis_ids", "keys", "error"),
    [
        pytest.param(
            [{"year": "2030"}], year_axes(), ["forecast"], ["2030"], None, id="calendar-2030"
        ),
        pytest.param(
            [{"phase_year": "1"}], year_axes(), ["forecast"], ["Y1"], None, id="phase-year-string"
        ),
        pytest.param(
            ["первый операционный год"],
            year_axes(),
            ["forecast"],
            ["Y1"],
            None,
            id="first-operational-phrase",
        ),
        pytest.param(
            [{"phase_year": 1}], two_first_years(), ["forecast"], [], "many", id="two-phase-years"
        ),
        pytest.param(
            [{"flag": "repayment"}], year_axes(), ["forecast"], ["R1"], None, id="repayment-flag"
        ),
        pytest.param(
            ["год погашения"], year_axes(), ["forecast"], ["R1"], None, id="repayment-phrase"
        ),
        pytest.param(
            [{"period_key": "missing"}], year_axes(), ["forecast"], [], "none", id="missing-key"
        ),
        pytest.param([{"year": "2030"}], year_axes(), ["other"], [], "none", id="other-axis"),
        pytest.param([], year_axes(), ["forecast"], [], None, id="empty-request"),
        pytest.param(
            [{"year": "Y1", "flag": "relevant"}],
            _PL,
            ["P&L!r2"],
            ["Y1"],
            None,
            id="unknown-flag-keeps-year",
        ),
        pytest.param(
            [{"year": "Y1", "period_key": "", "flag": ""}],
            _PL,
            ["P&L!r2"],
            ["Y1"],
            None,
            id="blank-fields",
        ),
        pytest.param([{"flag": "relevant"}], _PL, ["P&L!r2"], [], "none", id="unknown-flag-only"),
        pytest.param(
            [{"period_key": ""}, {"period_key": "Y10"}],
            _Y5_Y10,
            ["forecast"],
            ["Y10"],
            None,
            id="blank-spec-keeps-next",
        ),
        pytest.param(
            periods_from_question("Какой EBITDA в первый операционный год?", _TIMELINE),
            _TIMELINE,
            ["timeline"],
            ["2026"],
            None,
            id="construction-is-not-first-operation",
        ),
        pytest.param(
            periods_from_question("Какой Debt service в год начала погашения?", _SIBLING),
            _SIBLING,
            ["P&L!r2"],
            ["Y5"],
            None,
            id="sibling-repayment",
        ),
        pytest.param(
            [{"year": "2030"}], _ONLY_Y1, ["forecast"], [], "none", id="calendar-not-on-axis"
        ),
        pytest.param([{"year": "2030"}], _CALENDAR_KEY, [], [], "none", id="empty-axis-ids"),
        pytest.param(
            [{"year": "2030"}, {"period_key": "Y1"}],
            year_axes(),
            ["forecast"],
            ["2030", "Y1"],
            None,
            id="two-periods-keep-order",
        ),
    ],
)
def test_resolve_periods(wanted, axes, axis_ids, keys, error) -> None:
    found, got = resolve_periods(wanted, axes, axis_ids)
    assert found == keys
    assert got == error


@pytest.mark.parametrize(
    ("read", "question", "axes", "expected"),
    [
        pytest.param(
            periods_named_in_question,
            "Какой EBITDA в Y1?",
            _Y1_Y10,
            [{"period_key": "Y1"}],
            id="named-y1",
        ),
        pytest.param(
            periods_named_in_question,
            "Сравни Y10 и Y1",
            _Y1_Y10,
            [{"period_key": "Y10"}, {"period_key": "Y1"}],
            id="y1-not-inside-y10",
        ),
        pytest.param(
            periods_named_in_question,
            "Какой DSCR в 2030?",
            _Y1_Y2,
            [],
            id="calendar-not-read-from-y-key",
        ),
        pytest.param(
            periods_from_question,
            "Сравни EBITDA в Y5 и в Y10.",
            _Y5_Y10,
            [{"period_key": "Y5"}, {"period_key": "Y10"}],
            id="question-y5-and-y10",
        ),
        pytest.param(
            periods_from_question,
            "Какой EBITDA в 2030?",
            _ONLY_Y1,
            [{"year": "2030"}],
            id="calendar-year-requested",
        ),
    ],
)
def test_periods_read_from_the_question(read, question, axes, expected) -> None:
    assert read(question, axes) == expected
