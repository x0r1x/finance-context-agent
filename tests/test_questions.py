"""The closed set a choice is built from. No graph and no network."""

from finance_context_agent.questions import choice_rows, inventory_rows, ranker_state
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


def test_capex_inventory_lists_every_sheet() -> None:
    question = "какие еще есть CAPEX в книге и на каких листах?"
    rows = [
        catalog_row("exact", "CAPEX (including SPV costs)", sheet="Input Assumptions"),
        catalog_row("bare", "CAPEX", sheet="Construction"),
        catalog_row("bare-again", "CAPEX", sheet="Construction"),
        catalog_row("short", "CAPEX (incl. SPV costs)", sheet="Ratios"),
        catalog_row("other", "EBITDA", sheet="P&L"),
    ]
    shown, total = inventory_rows(question, rows)
    assert [(row["label"], row["sheet"]) for row in shown] == [
        ("CAPEX (including SPV costs)", "Input Assumptions"),
        ("CAPEX", "Construction"),
        ("CAPEX (incl. SPV costs)", "Ratios"),
    ]
    assert total == 3
    assert shown[1]["row_key"] == "bare"


def test_debt_inventory_is_wider_than_the_choice_pool() -> None:
    question = "перечисли строки про Debt"
    labels = (
        ("tba", "DEBT TIMELINE", "TBA"),
        ("amount", "Total Debt Amount (k£)", "Construction"),
        ("src", "Debt (k£)", "Construction"),
        ("sched", "Debt Drawdown Schedule", "Construction"),
        ("bop", "Debt Outstanding BoP (k£)", "Construction"),
        ("draw", "Debt Drawdown (k£)", "Construction"),
        ("eop", "Debt Outstanding EoP (k£)", "Construction"),
        ("repay", "Debt Repayment Schedule", "Debt"),
        ("service", "Debt service", "Debt"),
        ("cfads", "Cash Flow Available for Debt Service (CFADS)", "CFS"),
        ("bs", "Debt", "Balance Sheet"),
    )
    rows = [catalog_row(key, label, sheet=sheet) for key, label, sheet in labels]
    shown, total = inventory_rows(question, rows)
    assert total == 11
    assert [row["row_key"] for row in shown] == [key for key, _label, _sheet in labels]
    assert len(choice_rows(question, rows)) == 8


def test_inventory_stops_at_twenty_four() -> None:
    rows = [catalog_row(f"c{index}", f"CAPEX {index}") for index in range(25)]
    shown, total = inventory_rows("какие есть CAPEX", rows)
    assert total == 25
    assert len(shown) == 24
    assert shown[0]["row_key"] == "c0"
    assert shown[-1]["row_key"] == "c23"


def test_inventory_ignores_the_filename_and_unrelated_labels() -> None:
    rows = [
        catalog_row("cover", "PROJECT FINANCING", sheet="Input Assumptions"),
        catalog_row("other", "EBITDA", sheet="P&L"),
    ]
    assert inventory_rows(f"сделай саммари {_FILE}", rows, [_FILE]) == ([], 0)


def test_ranker_state_names_the_book_only_when_no_row_is_offered() -> None:
    uttered = ranker_state(_FILE, "Debt Outstanding BoP", True)
    assert not uttered.startswith("Книга ")
    assert uttered == "Debt Outstanding BoP"
    empty = ranker_state(_FILE, "сделай саммари", False)
    assert empty.startswith(f"Книга {_FILE}.")
