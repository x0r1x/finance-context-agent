"""The closed set a choice is built from. No graph and no network."""

from finance_context_agent.questions import choice_rows, ranker_state
from tests.fakes import catalog_row

_FILE = "packt-project-finance.xlsx"
_DEBT = (
    ("tba", "DEBT TIMELINE", "TBA"),
    ("src", "Debt (k£)", "Construction"),
    ("sched", "Debt Drawdown Schedule", "Construction"),
    ("bop", "Debt Outstanding BoP (k£)", "Construction"),
    ("draw", "Debt Drawdown (k£)", "Construction"),
    ("eop", "Debt Outstanding EoP (k£)", "Construction"),
    ("repay", "Debt Repayment Schedule", "Debt"),
    ("service", "Debt service", "Debt"),
    ("bs", "Debt", "Balance Sheet"),
)


def _debt_rows() -> list[dict]:
    return [catalog_row(key, label, sheet=sheet) for key, label, sheet in _DEBT]


def test_summary_with_a_filename_offers_no_row() -> None:
    rows = [
        catalog_row("cover", "PROJECT FINANCING", sheet="Input Assumptions"),
        catalog_row("irr-a", "Project IRR", sheet="Ratios"),
        catalog_row("irr-b", "Project IRR", sheet="Ratios"),
    ]
    assert choice_rows(f"сделай саммари {_FILE}", rows, [_FILE]) == []


def test_debt_phrase_keeps_bop_and_eop() -> None:
    rows = _debt_rows()
    bare = [row["label"] for row in choice_rows("Debt Outstanding BoP", rows)]
    with_unit = [row["label"] for row in choice_rows("Debt Outstanding BoP (k£)", rows)]
    assert bare == ["Debt Outstanding BoP (k£)", "Debt Outstanding EoP (k£)"]
    assert with_unit == bare


def test_capex_phrase_keeps_the_exact_line_and_the_shorter_one() -> None:
    exact = "CAPEX (including SPV costs)"
    heavy = "CAPEX (including heavy maintenance & SPV costs)"
    short = "CAPEX (incl. SPV costs)"
    rows = [
        catalog_row("exact", exact),
        catalog_row("heavy", heavy),
        catalog_row("short", short),
    ]
    labels = [
        row["label"]
        for row in choice_rows("расскажи про CAPEX (including SPV costs) по годам", rows)
    ]
    assert labels == [exact, heavy, short]


def test_debt_service_does_not_pull_the_score_one_lines() -> None:
    labels = [row["label"] for row in choice_rows("Какой Debt service в Y1?", _debt_rows())]
    assert labels == ["Debt service"]


def test_a_specific_long_label_does_not_keep_the_shorter_copy() -> None:
    minimum = "Minimum Debt Service Coverage Ratio (DSCR)"
    average = "Average Debt Service Coverage Ratio (DSCR)"
    generic = "Debt Service Coverage Ratio (DSCR)"
    rows = [
        catalog_row("min", minimum),
        catalog_row("avg", average),
        catalog_row("raw", generic),
    ]
    labels = [row["label"] for row in choice_rows(f"Какой {minimum}?", rows)]
    assert labels == [minimum]


def test_ebitda_question_keeps_both_rows() -> None:
    short = "EBITDA"
    long = "Operating Income or Loss (EBITDA)"
    rows = [
        catalog_row("short", short, sheet="PF Model"),
        catalog_row("long", long, sheet="PF Model"),
        catalog_row("capex", "CAPEX"),
    ]
    labels = [row["label"] for row in choice_rows("Какой EBITDA?", rows)]
    assert labels == [short, long]
    assert [row["label"] for row in choice_rows("Какой EBITDA в 2030?", rows)] == labels


def test_ranker_state_names_the_book_only_when_no_row_is_offered() -> None:
    uttered = ranker_state(_FILE, "Debt Outstanding BoP", True)
    assert not uttered.startswith("Книга ")
    assert uttered == "Debt Outstanding BoP"
    empty = ranker_state(_FILE, "сделай саммари", False)
    assert empty.startswith(f"Книга {_FILE}.")
