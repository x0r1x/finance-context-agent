import json

from finance_context_agent.prompts import answer_messages, plan_messages, view_messages
from tests.fakes import observation


def test_answer_prompt_drops_foreign_rows_and_context_json() -> None:
    system, user = answer_messages(
        {
            "question": "Какой DSCR?",
            "observations": [observation("row-dscr", "2030", "1.25", "C10")],
            "citations": [
                {"row_key": "row-foreign", "period_id": "2030", "value": "5", "cell": "Z1"},
                {"row_key": "row-dscr", "period_id": "2030", "value": "1.25", "cell": "C10"},
            ],
            "gaps": [],
        }
    )
    assert "row-foreign" not in user
    assert "row-dscr" in user
    assert "context.json" not in system
    assert "context.json" not in user


def test_plan_prompt_has_axes_and_not_a_catalog_page() -> None:
    system, user = plan_messages(
        {
            "question": "Какой DSCR?",
            "human_reply": "наблюдённый",
            "summary": {"marker": "passport"},
            "axes": [{"id": "forecast", "periods": [{"period_key": "2030"}]}],
            "citations": [{"row_key": "row-dscr", "label": "DSCR", "period_id": "2030"}],
            "gaps": ["precedent"],
            "rows": [{"row_key": "SHOULD_NOT_LEAK", "page_marker": "CATALOG_PAGE"}],
        }
    )
    assert "passport" in user
    assert "2030" in user
    assert "row-dscr" in user
    assert "precedent" in user
    assert "CATALOG_PAGE" not in user
    assert "SHOULD_NOT_LEAK" not in user
    assert "context.json" not in system + user
    assert "не в needles" in system
    assert "без слова «какой»" in system
    assert "почему или из чего" in system


def test_answer_prompt_omits_the_account_code() -> None:
    observed = observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    observed["concept_id"] = "cov.dscr"
    observed["formula"] = {
        "precedents": [{"row_key": "prec", "concept_id": "cf.cfads", "label": "CFADS"}]
    }
    _system, user = answer_messages(
        {"question": "Какой DSCR?", "observations": [observed], "citations": [], "gaps": []}
    )
    assert "concept_id" not in user
    assert "cov.dscr" not in user
    assert "cf.cfads" not in user
    assert "DSCR" in user
    assert "1.25" in user


def test_view_prompt_describes_the_layouts_and_carries_only_the_summary() -> None:
    system, user = view_messages(
        {
            "question": "ряд",
            "points": 2,
            "labels": ["A"],
            "periods": ["Y1", "Y2"],
            "statuses": ["cached"],
            "scales": [],
            "explain": False,
        }
    )
    assert system == (
        "Выбери вид уже собранного кадра. sentence — каждая точка отдельной фразой. "
        "table-period — строки это периоды, столбцы это подписи. "
        "table-label — строки это подписи, столбцы это периоды. "
        "Несколько точек одной подписи удобнее таблицей. "
        "Если точка одна и вопрос не просит разложить точки, выбери sentence. "
        "Числа не пиши."
    )
    assert json.loads(user)["question"] == "ряд"
    assert "36.5" not in system
