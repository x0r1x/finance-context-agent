import pytest

from finance_context_agent.clients.llm import ModelError
from finance_context_agent.clients.ranker import _choice, _sheets

_CRITERIA = {"billing": "Charges and invoices", "technical": "Bugs and outages"}
_PICK = {"choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.1}}


def test_official_answers_key_is_a_choice() -> None:
    choice = _choice({"model": "clm-latest", "answers": {"pick": _PICK}}, _CRITERIA)
    assert choice.key == "billing"
    assert choice.probabilities == {"billing": 0.9, "technical": 0.1}


def test_questions_key_still_wins_when_both_are_present() -> None:
    body = {
        "questions": {"pick": _PICK},
        "answers": {
            "pick": {
                "choice": "technical",
                "probabilities": {"billing": 0.1, "technical": 0.9},
            }
        },
    }
    assert _choice(body, _CRITERIA).key == "billing"


def test_a_body_without_either_key_is_rejected() -> None:
    with pytest.raises(ModelError, match="ranker schema"):
        _choice({"model": "clm-latest"}, _CRITERIA)


def test_sheets_score_is_read_beside_the_pick() -> None:
    body = {
        "questions": {
            "pick": _PICK,
            "act": {"probabilities": {"values": 0.109, "sheets": 0.891}},
        }
    }
    choice = _choice(body, _CRITERIA)
    choice.sheets = _sheets(body)
    assert choice.key == "billing"
    assert choice.sheets == 0.891


def test_questions_key_wins_for_the_act_too() -> None:
    body = {
        "questions": {"act": {"probabilities": {"sheets": 0.891, "values": 0.109}}},
        "answers": {"act": {"probabilities": {"sheets": 0.1, "values": 0.9}}},
    }
    assert _sheets(body) == 0.891


def test_a_pick_without_an_act_is_rejected_by_the_sheets_reader() -> None:
    with pytest.raises(ModelError, match="ranker schema"):
        _sheets({"questions": {"pick": _PICK}})
