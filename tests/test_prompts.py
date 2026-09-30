from finance_context_agent.prompts import answer_messages, plan_messages
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
