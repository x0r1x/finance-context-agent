"""The closed set a choice is built from. No graph and no network."""

from finance_context_agent.questions import (
    ACT_FLOOR,
    ACT_GAP,
    ACTS,
    catalog_reply,
    choice_rows,
    class_scores,
    goal_question,
    inventory_rows,
    match_goal,
    named_label_rows,
    prototype_decision,
    ranker_state,
)
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


def test_named_label_is_that_row() -> None:
    rows = _debt_rows() + [
        catalog_row("cfads", "Cash Flow Available for Debt Service (CFADS)", sheet="CFS"),
        catalog_row("amount", "Total Debt Amount (k£)", sheet="Construction"),
        catalog_row("conc", "Concession Duration", sheet="Input Assumptions"),
        catalog_row("build", "Construction Duration", sheet="Input Assumptions"),
        catalog_row("ops", "Operations Duration", sheet="Input Assumptions"),
        catalog_row("infl", "Inflation per year (costs) from beginning of concession"),
        catalog_row("end", "Cash out End of Concession", sheet="Ratios"),
        catalog_row("irr-a", "Project IRR", sheet="Ratios"),
        catalog_row("irr-b", "Project IRR", sheet="Cover"),
    ]
    assert [row["row_key"] for row in named_label_rows("Debt service", rows) or []] == ["service"]
    assert [row["row_key"] for row in named_label_rows("Debt Outstanding BoP", rows) or []] == [
        "bop"
    ]
    assert [row["row_key"] for row in named_label_rows("Concession Duration", rows) or []] == [
        "conc"
    ]
    assert [row["row_key"] for row in named_label_rows("Project IRR", rows) or []] == [
        "irr-a",
        "irr-b",
    ]
    named = named_label_rows(f"Debt service {_FILE}", rows, [_FILE])
    assert [row["row_key"] for row in named or []] == ["service"]


def test_a_question_or_a_shared_word_is_not_a_named_label() -> None:
    rows = _debt_rows() + [
        catalog_row("cfads", "Cash Flow Available for Debt Service (CFADS)", sheet="CFS"),
        catalog_row("long", "CAPEX (including SPV costs)", sheet="Input Assumptions"),
        catalog_row("bare", "CAPEX", sheet="Construction"),
        catalog_row("short", "CAPEX (incl. SPV costs)", sheet="Ratios"),
        catalog_row("pl", "P&L", sheet="P&L"),
    ]
    for text in (
        "на каком листе Debt service",
        "перечисли строки про Debt",
        "Какой Debt service в Y1?",
        "CAPEX",
        "Debt",
        "Debt (k£)",
        "P&L",
    ):
        assert named_label_rows(text, rows) is None


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


def _capex_book() -> list[dict]:
    return [
        catalog_row(
            "exact",
            "CAPEX (including SPV costs)",
            sheet="Input Assumptions",
            label_path=["COSTS DURING CONSTRUCTION"],
        ),
        catalog_row("bare", "CAPEX", sheet="Construction"),
        catalog_row(
            "short",
            "CAPEX (incl. SPV costs)",
            sheet="Ratios",
            label_path=["Project IRR"],
        ),
        catalog_row("ebitda", "EBITDA", sheet="P&L"),
    ]


def test_prototype_decision_needs_both_floors() -> None:
    scores = {name: ACT_FLOOR - ACT_GAP for name in ACTS}
    scores["catalog"] = ACT_FLOOR
    decided = prototype_decision(scores)
    assert decided is not None
    act, top, gap = decided
    assert act == "catalog"
    assert top == ACT_FLOOR
    assert abs(gap - ACT_GAP) < 1e-9
    low = dict(scores)
    low["catalog"] = ACT_FLOOR - 0.001
    assert prototype_decision(low) is None
    narrow = {name: ACT_FLOOR - ACT_GAP + 0.001 for name in ACTS}
    narrow["catalog"] = ACT_FLOOR
    assert prototype_decision(narrow) is None
    assert prototype_decision({name: 0.01 for name in ACTS}) is None
    assert prototype_decision(None) is None


def test_max_keeps_a_matching_anchor_and_the_centroid_shrinks_it() -> None:
    query = [1.0, 0.0]
    anchors = {"catalog": [[1.0, 0.0], [0.0, 1.0]]}
    assert class_scores(query, anchors, how="max")["catalog"] == 1.0
    assert class_scores(query, anchors, how="centroid")["catalog"] < 1.0


_CATALOG_TABLE = (
    "Лист Input Assumptions\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX (including SPV costs) | COSTS DURING CONSTRUCTION |\n"
    "\n"
    "Лист Construction\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX | |\n"
    "\n"
    "Лист Ratios\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX (incl. SPV costs) | Project IRR |\n"
    "\n"
    "Лист P&L\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| EBITDA | |"
)

_CONSTRUCTION_TABLE = (
    "Лист Construction\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX | |"
)


def test_catalog_reply_groups_sheets_and_can_keep_one() -> None:
    rows = _capex_book()
    full = catalog_reply(rows, "дай список атрибутов в этой книге")
    assert full == _CATALOG_TABLE
    construction = catalog_reply(rows, "что есть на листе Construction")
    assert construction == _CONSTRUCTION_TABLE
    assert "EBITDA" not in construction
    assert "Input Assumptions" not in construction


def test_catalog_table_escapes_a_pipe_and_joins_a_broken_label() -> None:
    piped = catalog_reply(
        [catalog_row("debt", "A | B", sheet="Debt", label_path=["X"])],
        "",
    )
    assert "| A \\| B | X |" in piped
    assert "A | B" not in piped
    assert piped.count("| --- | --- |") == 1
    broken = catalog_reply([catalog_row("split", "A\nB", sheet="Debt")], "")
    assert "| A B | |" in broken
    data_rows = [
        line
        for line in broken.splitlines()
        if line.startswith("| ") and not line.startswith("| Атрибут") and set(line) != set("| -")
    ]
    assert data_rows == ["| A B | |"]
    assert catalog_reply([], "") == ""


def test_goal_question_names_the_book_and_four_actions() -> None:
    asked = goal_question(_FILE)
    assert asked.startswith(f"Книга {_FILE} открыта.")
    assert "Что посмотреть?" in asked
    assert "1. Обзор книги" in asked
    assert "2. Перечень атрибутов" in asked
    assert "3. Одну метрику" in asked
    assert "4. Другой файл" in asked
    assert goal_question("  ").startswith("Книга открыта.")
    accepted = {
        "1": "overview",
        "№1": "overview",
        "1.": "overview",
        "обзор книги": "overview",
        "1. обзор книги": "overview",
        "2": "catalog",
        "№2": "catalog",
        "2.": "catalog",
        "перечень атрибутов": "catalog",
        "2. перечень атрибутов": "catalog",
        "3": "figure",
        "№3": "figure",
        "3.": "figure",
        "одну метрику": "figure",
        "3. одну метрику": "figure",
        "4": "files",
        "№4": "files",
        "4.": "files",
        "другой файл": "files",
        "4. другой файл": "files",
    }
    for reply, act in accepted.items():
        assert match_goal(reply) == act
        assert match_goal(f"  {reply.upper()}  ") == act
    for reply in ("", "   ", "обзор", "давай номер 1", "открой", "первый", "Какой EBITDA в Y1?"):
        assert match_goal(reply) is None
