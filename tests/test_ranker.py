import pytest

from finance_context_agent.clients.llm import ModelError
from finance_context_agent.clients.ranker import _choice

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
