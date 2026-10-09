"""How a cached frame is shown. The model picks the view; these tests pick it directly."""

import json

import pytest

from finance_context_agent.draft import frame_summary, legal_views, render_frame
from finance_context_agent.questions import _pipe_table
from tests.fakes import observation

_LABEL = "Concession Duration"
_TABLE = (
    "| Период | Concession Duration |\n"
    "| --- | --- |\n"
    "| | 40 |\n"
    "| base-case | 40 |\n"
    "| scenario-2 | 36.5 |\n"
    "| gearing | 38 |\n"
    "| scenario-4 | 35.5 |"
)
_SENTENCES = "\n".join(
    [
        "Concession Duration: 40",
        "Concession Duration в base-case: 40",
        "Concession Duration в scenario-2: 36.5",
        "Concession Duration в gearing: 38",
        "Concession Duration в scenario-4: 35.5",
    ]
)


def _series() -> list[dict]:
    points = [
        ("value", "40"),
        ("base-case", "40"),
        ("scenario-2", "36.5"),
        ("gearing", "38"),
        ("scenario-4", "35.5"),
    ]
    rows = [
        observation("in|duration", period, value, f"C{index}", label=_LABEL)
        for index, (period, value) in enumerate(points)
    ]
    rows[0]["concept_id"] = "time.concession"
    rows[0]["formula"] = {"text": "=B1", "precedents": []}
    return rows


def _summary(rows: list[dict], question: str = _LABEL) -> dict:
    return frame_summary(
        {"question": question, "plan": {"question_type": "lookup", "trace": "none"}},
        rows,
    )


def _pipes(line: str) -> int:
    count = 0
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            count += 1
    return count


def test_concession_duration_table_and_sentence_share_one_frame() -> None:
    rows = _series()
    assert render_frame(rows, "table-period") == _TABLE
    assert render_frame(rows, "sentence") == _SENTENCES
    assert "в value" not in _TABLE
    summary = _summary(rows)
    assert set(summary) == {
        "question",
        "points",
        "labels",
        "periods",
        "statuses",
        "scales",
        "explain",
    }
    blob = json.dumps(summary, ensure_ascii=False)
    assert "36.5" not in blob
    assert "concept_id" not in blob
    assert "=B1" not in blob
    assert summary["periods"] == ["", "base-case", "scenario-2", "gearing", "scenario-4"]
    assert "table-label" not in legal_views(summary, rows)


def test_a_pipe_in_the_label_does_not_split_the_row() -> None:
    rows = [
        observation("row", "Y1", "1", "C1", label="A | B"),
        observation("row", "Y2", "2", "C2", label="A | B"),
    ]
    text = render_frame(rows, "table-period")
    assert "A \\| B" in text
    assert "A | B" not in text
    widths = [_pipes(line) for line in text.splitlines()]
    assert len(set(widths)) == 1
    assert widths[0] == 3


def test_an_empty_cell_is_not_zero_and_the_scale_stays_in_the_cell() -> None:
    empty = observation("row", "Y1", "", "C1", status="empty", label="DSCR")
    filled = observation("row", "Y2", "1.25", "C2", label="DSCR")
    text = render_frame([empty, filled], "table-period")
    assert "| Y1 | пусто |" in text
    assert "0" not in text
    scaled = observation(
        "row", "Y40", "10", "C3", scale_factor=1000, scale="k", label="EBITDA"
    )
    shown = render_frame([scaled], "table-period")
    assert "| Y40 | 10 тыс. |" in shown


def test_a_missing_intersection_stays_blank_and_years_keep_book_order() -> None:
    rows = [
        observation("a", "scenario-2", "1", "A1", label="EBITDA"),
        observation("b", "scenario-2", "3", "B1", label="DSCR"),
        observation("a", "base-case", "2", "A2", label="EBITDA"),
    ]
    text = render_frame(rows, "table-label")
    assert text.index("| scenario-2 |") < text.index("| base-case |")
    assert "| DSCR | 3 | |" in text
    assert "пусто" not in text
    assert "| 0 |" not in text


def test_one_nameless_point_is_not_a_table_and_four_keys_still_are() -> None:
    lone = [observation("row", "value", "40", "B1", label=_LABEL)]
    assert legal_views(_summary(lone), lone) == ["sentence"]
    named = [observation("row", "Y1", "0", "C1", status="zero_explicit", label="EBITDA")]
    assert legal_views(_summary(named), named) == ["sentence", "table-period", "table-label"]
    four = [
        observation("row", period, "1", "C1", label=_LABEL)
        for period in ("", "base-case", "scenario-2", "gearing")
    ]
    assert "table-label" in legal_views(_summary(four), four)
    repeated = [
        observation("row", "base-case", "40", "C1", label=_LABEL),
        observation("row", "base-case", "41", "C9", label=_LABEL),
    ]
    assert legal_views(_summary(repeated), repeated) == ["sentence"]


def test_formulas_sit_under_the_table_and_a_single_sentence_keeps_its_line() -> None:
    first = observation("row", "Y1", "1.10", "C1", label="DSCR")
    first["formula"] = {"text": "=H10/I10", "precedents": []}
    second = observation("row", "Y2", "1.25", "C2", label="DSCR")
    second["formula"] = {"text": "=I10/J10", "precedents": []}
    text = render_frame([first, second], "table-period", explain=True)
    head, tail = text.split("\n\n", 1)
    assert head.startswith("| Период |")
    assert tail.startswith("Y1\nФормула C1: =H10/I10")
    assert "Формула C2: =I10/J10" in tail
    assert "\n\n" not in tail
    alone = render_frame([first], "sentence", explain=True)
    assert alone.startswith("DSCR в Y1: 1.10\nФормула C1: =H10/I10")
    assert "| Период |" not in alone


def test_a_short_table_row_is_not_padded() -> None:
    with pytest.raises(ValueError, match="cells"):
        _pipe_table(["Период", "DSCR"], [["Y1"]])
