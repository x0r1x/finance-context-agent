"""Fields a new user turn overwrites. Book passport, axes and citations stay."""

from __future__ import annotations

from typing import Any

from finance_context_agent.settings import Settings, field_default


def new_turn_input(question: str, job_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "human_reply": "",
        "search_misses": 0,
        "content_steps": 0,
        "gaps": [],
        "gap_keys": [],
        "user_question": "",
        "observations": [],
        "awaiting": "",
        "pending": "",
        "terminal": "",
        "satisfactory": False,
        "same_gap": False,
        "reload": False,
        "search_again": False,
        "just_bound_job": False,
        "just_bound_row": False,
        "offers": [],
        "clarify_rounds": 0,
        "last_user_question": "",
        "menu_question": "",
        "steps": [],
        "draft": "",
        "search_note": "",
        "plan": {},
        "proposed_citations": [],
        "schema_error": False,
    }
    if job_id:
        payload["job_id"] = job_id
    return payload


def run_config(thread_id: str, settings: Settings | None = None) -> dict[str, Any]:
    limit = settings.recursion_limit if settings is not None else field_default("recursion_limit")
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": limit}
