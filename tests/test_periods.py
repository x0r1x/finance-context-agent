from finance_context_agent.periods import periods_named_in_question, resolve_periods
from tests.fakes import two_first_years, year_axes


def test_calendar_year_becomes_one_period_key() -> None:
    keys, error = resolve_periods([{"year": "2030"}], year_axes(), ["forecast"])
    assert error is None
    assert keys == ["2030"]


def test_phase_year_string_matches() -> None:
    keys, error = resolve_periods([{"phase_year": "1"}], year_axes(), ["forecast"])
    assert keys == ["Y1"]
    assert error is None


def test_first_operational_year_phrase() -> None:
    keys, error = resolve_periods(["первый операционный год"], year_axes(), ["forecast"])
    assert keys == ["Y1"]
    assert error is None


def test_two_phase_years_are_many() -> None:
    keys, error = resolve_periods([{"phase_year": 1}], two_first_years(), ["forecast"])
    assert keys == []
    assert error == "many"


def test_repayment_flag() -> None:
    keys, error = resolve_periods([{"flag": "repayment"}], year_axes(), ["forecast"])
    assert keys == ["R1"]
    assert error is None


def test_phrase_repayment() -> None:
    keys, error = resolve_periods(["год погашения"], year_axes(), ["forecast"])
    assert keys == ["R1"]
    assert error is None


def test_missing_period_is_none() -> None:
    keys, error = resolve_periods([{"period_key": "missing"}], year_axes(), ["forecast"])
    assert keys == []
    assert error == "none"


def test_other_axis_hides_the_period() -> None:
    keys, error = resolve_periods([{"year": "2030"}], year_axes(), ["other"])
    assert error == "none"
    assert keys == []


def test_empty_request_keeps_the_whole_series() -> None:
    keys, error = resolve_periods([], year_axes(), ["forecast"])
    assert keys == []
    assert error is None


def test_unknown_flag_does_not_veto_a_year_key() -> None:
    axes = [{"id": "P&L!r2", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}]
    keys, error = resolve_periods([{"year": "Y1", "flag": "relevant"}], axes, ["P&L!r2"])
    assert error is None
    assert keys == ["Y1"]


def test_blank_period_fields_do_not_veto() -> None:
    axes = [{"id": "P&L!r2", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}]
    keys, error = resolve_periods(
        [{"year": "Y1", "period_key": "", "flag": ""}],
        axes,
        ["P&L!r2"],
    )
    assert keys == ["Y1"]
    assert error is None


def test_only_an_unknown_flag_matches_nothing() -> None:
    axes = [{"id": "P&L!r2", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}]
    keys, error = resolve_periods([{"flag": "relevant"}], axes, ["P&L!r2"])
    assert keys == []
    assert error == "none"


def test_named_period_key_survives_an_empty_planner_period() -> None:
    axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}, {"period_key": "Y10"}]}]
    assert periods_named_in_question("Какой EBITDA в Y1?", axes) == [{"period_key": "Y1"}]


def test_y1_does_not_match_inside_y10() -> None:
    axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}, {"period_key": "Y10"}]}]
    assert periods_named_in_question("Сравни Y10 и Y1", axes) == [
        {"period_key": "Y10"},
        {"period_key": "Y1"},
    ]


def test_calendar_year_is_not_taken_from_a_y_key() -> None:
    axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}]
    assert periods_named_in_question("Какой DSCR в 2030?", axes) == []


def test_two_requested_periods_stay_in_order() -> None:
    keys, error = resolve_periods(
        [{"year": "2030"}, {"period_key": "Y1"}],
        year_axes(),
        ["forecast"],
    )
    assert keys == ["2030", "Y1"]
    assert error is None
