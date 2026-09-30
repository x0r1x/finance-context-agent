from types import SimpleNamespace

import pytest
from langgraph.types import Command

from finance_context_agent.session import JobMismatchError, decide_input
from finance_context_agent.turn import run_config


def _snap(*, values=None, nxt=(), interrupts=()):
    return SimpleNamespace(values=values or {}, next=nxt, interrupts=interrupts, tasks=())


def test_run_config_keeps_recursion_limit_beside_configurable() -> None:
    config = run_config("thread-1")
    assert config["recursion_limit"] == 40
    assert "recursion_limit" not in config["configurable"]
    assert config["configurable"]["thread_id"] == "thread-1"


def test_interrupt_resumes_with_the_user_text() -> None:
    payload = decide_input(
        _snap(values={"job_id": "job-1"}, interrupts=["which row?"]), "наблюдённый", "job-1"
    )
    assert isinstance(payload, Command)
    assert payload.resume == "наблюдённый"


def test_pending_node_continues_without_a_new_question() -> None:
    payload = decide_input(
        _snap(values={"job_id": "job-1"}, nxt=("answer",)), "другой вопрос", "job-1"
    )
    assert payload is None


def test_finished_thread_starts_a_new_turn() -> None:
    payload = decide_input(
        _snap(values={"job_id": "job-1", "citations": [{"row_key": "r"}]}), "А почему?", "job-1"
    )
    assert payload["question"] == "А почему?"
    assert "citations" not in payload
    assert payload["job_id"] == "job-1"


def test_different_job_on_a_bound_thread_is_rejected() -> None:
    with pytest.raises(JobMismatchError):
        decide_input(_snap(values={"job_id": "job-1"}), "вопрос", "job-2")


def test_empty_snapshot_accepts_the_request_job() -> None:
    payload = decide_input(_snap(), "вопрос", "job-1")
    assert payload["job_id"] == "job-1"
    assert payload["question"] == "вопрос"
