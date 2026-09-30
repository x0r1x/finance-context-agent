"""Offline checks for the business-answer scorer. No network."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_PATH = ROOT / "scripts" / "check-answers.py"
_SPEC = importlib.util.spec_from_file_location("check_answers", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_answers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_answers)


def test_scalar_lookup_reads_the_value_column(monkeypatch) -> None:
    payload = {
        "observations": [
            {
                "row_key": "PF|53",
                "period_id": "value",
                "label": "Average Debt Service Coverage Ratio (DSCR)",
                "value": "1.8617377551507139",
                "value_status": "cached",
            }
        ]
    }
    monkeypatch.setattr(check_answers, "_request_json", lambda *_args: (200, payload))
    found = check_answers._observation("http://parser", "job", "PF|53", "")
    assert found is not None
    assert found["value"] == "1.8617377551507139"


def test_scalar_lookup_does_not_take_a_single_year(monkeypatch) -> None:
    payload = {"observations": [{"row_key": "P&L|13", "period_id": "Y5", "value": "1"}]}
    monkeypatch.setattr(check_answers, "_request_json", lambda *_args: (200, payload))
    assert check_answers._observation("http://parser", "job", "P&L|13", "") is None


def test_case_file_names_the_two_books_and_unique_ids() -> None:
    document = check_answers.load_cases(ROOT / "scripts" / "business_cases.json")
    cases = document["cases"]
    ids = [item["id"] for item in cases]
    assert len(ids) == len(set(ids))
    assert {item["book"] for item in cases} == {
        "packt-project-finance.xlsx",
        "rvi-project-finance.xlsx",
    }
    y1 = next(item for item in cases if item["id"] == "packt-ebitda-y1")
    assert y1["expect"]["period_id"] == "Y1"
    assert y1["expect"]["value_status"] == "zero_explicit"
    assert y1["expect"]["forbid_periods"] == ["Y10"]
    assert {item["id"] for item in cases if item["group"] == "phase"} == {
        "packt-ebitda-first-operation",
        "rvi-ebitda-first-operation",
        "packt-debt-service-repayment-start",
    }
    skipped = {item["file"] for item in document["skipped"]}
    assert "АШАН_модель CF.xls" in skipped


def test_explicit_zero_citation_is_green_without_a_scale_word() -> None:
    case = _citation("packt-ebitda-y1", "Y1", "zero_explicit")
    completion = _completion(
        "0",
        [_cite("P&L|13|P&L!r2", "Y1", "0", "zero_explicit")],
    )
    result = check_answers.score_case(case, completion, _only(_zero_obs()))
    assert result["attribute_ok"] is True
    assert result["value_ok"] is True
    assert result["wording_ok"] is True
    assert result["passed"] is True


def test_y10_citation_on_a_y1_question_is_red() -> None:
    case = _citation("packt-ebitda-y1", "Y1", "zero_explicit")
    case["expect"]["forbid_periods"] = ["Y10"]
    completion = _completion(
        "90099",
        [_cite("P&L|13|P&L!r2", "Y10", "90099", "cached")],
    )

    def observation_for(row_key: str, period_id: str) -> dict[str, Any] | None:
        if row_key == "P&L|13|P&L!r2" and period_id == "Y10":
            return _cached_obs("Y10", "90099")
        return None

    result = check_answers.score_case(case, completion, observation_for)
    assert result["attribute_ok"] is False
    assert result["value_ok"] is False
    assert result["passed"] is False


def test_dscr_pause_without_a_citation_is_green() -> None:
    case = {
        "id": "packt-dscr",
        "book": "packt-project-finance.xlsx",
        "group": "menu",
        "question": "Какой DSCR?",
        "expect": {"outcome": "ask"},
    }
    completion = _completion(
        "Такой строки нет. Назовите подпись иначе или выберите другую метрику.",
        [],
        awaiting=True,
    )
    result = check_answers.score_case(case, completion, lambda _row, _period: None)
    assert result["passed"] is True
    assert result["citations"] == []


def test_menu_count_is_not_an_invented_number() -> None:
    case = {
        "id": "rvi-ebitda-menu-2030",
        "book": "rvi-project-finance.xlsx",
        "group": "menu",
        "question": "Какой EBITDA в 2030?",
        "expect": {
            "outcome": "ask",
            "menu_contains": [
                "Operating Income or Loss (EBITDA)",
                "EBITDA [PF Model",
            ],
        },
    }
    content = (
        "Какую строку взять?\n"
        "«EBITDA»: 2. Operating Income or Loss (EBITDA) [PF Model, pnl.ebitda]; "
        "EBITDA [PF Model, pnl.ebitda]"
    )
    clean = check_answers.score_case(
        case, _completion(content, [], awaiting=True), lambda _row, _period: None
    )
    assert clean["passed"] is True
    invented = check_answers.score_case(
        case,
        _completion(content + " 7694.41", [], awaiting=True),
        lambda _row, _period: None,
    )
    assert invented["passed"] is False
    assert invented["value_ok"] is False
    twelve = (
        "Какую строку взять?\n"
        "«CFADS»: 8. Debt Service Coverage Ratio (last 12 months) "
        "[PF Model, cf.cfads]; CFADS [PF Model, cf.cfads]"
    )
    masked = check_answers.score_case(
        {
            "id": "rvi-cfads-2030",
            "book": "rvi-project-finance.xlsx",
            "group": "menu",
            "question": "Какой CFADS в 2030?",
            "expect": {"outcome": "ask"},
        },
        _completion(twelve, [], awaiting=True),
        lambda _row, _period: None,
    )
    assert masked["passed"] is True
    assert masked["value_ok"] is True


def test_missing_scale_word_keeps_the_cell_green() -> None:
    case = _citation("packt-ebitda-y5", "Y5", "cached")
    value = "66092.838205122098"
    completion = _completion(value, [_cite("P&L|13|P&L!r2", "Y5", value, "cached")])
    result = check_answers.score_case(case, completion, _only(_cached_obs("Y5", value)))
    assert result["attribute_ok"] is True
    assert result["value_ok"] is True
    assert result["wording_ok"] is False
    assert result["passed"] is True


def _citation(case_id: str, period_id: str, status: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "book": "packt-project-finance.xlsx",
        "group": "point",
        "question": f"Какой EBITDA в {period_id}?",
        "expect": {
            "outcome": "citation",
            "row_key": "P&L|13|P&L!r2",
            "period_id": period_id,
            "value_status": status,
        },
    }


def _completion(
    content: str,
    citations: list[dict[str, Any]],
    *,
    awaiting: bool = False,
) -> dict[str, Any]:
    return {
        "awaiting_user": awaiting,
        "gaps": [],
        "steps": [],
        "choices": [{"message": {"content": content}}],
        "citations": citations,
    }


def _cite(row_key: str, period_id: str, value: str, status: str) -> dict[str, Any]:
    return {
        "row_key": row_key,
        "period_id": period_id,
        "cell": "F13",
        "value": value,
        "value_status": status,
    }


def _zero_obs() -> dict[str, Any]:
    return {
        "row_key": "P&L|13|P&L!r2",
        "period_id": "Y1",
        "label": "EBITDA",
        "value": "0",
        "value_status": "zero_explicit",
        "normalized_value": "0",
        "scale_factor": 1000,
        "unit": {"scale": "k"},
    }


def _cached_obs(period_id: str, value: str) -> dict[str, Any]:
    return {
        "row_key": "P&L|13|P&L!r2",
        "period_id": period_id,
        "label": "EBITDA",
        "value": value,
        "value_status": "cached",
        "normalized_value": value + "000",
        "scale_factor": 1000,
        "unit": {"scale": "k"},
    }


def _only(observation: dict[str, Any]) -> Any:
    def observation_for(row_key: str, period_id: str) -> dict[str, Any] | None:
        if row_key == observation["row_key"] and period_id == observation["period_id"]:
            return observation
        return None

    return observation_for
