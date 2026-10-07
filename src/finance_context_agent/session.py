"""Choose the next graph input from a checkpoint snapshot."""

from __future__ import annotations

import time
from hashlib import sha256
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from finance_context_agent.turn import new_turn_input

AGENT_MODEL = "finance-context-agent"


class JobMismatchError(Exception):
    """The thread is already bound to a different workbook."""


def interrupts_of(snapshot: Any) -> list[Any]:
    raw = getattr(snapshot, "interrupts", None)
    if raw:
        return list(raw)
    found: list[Any] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        found.extend(list(getattr(task, "interrupts", ()) or ()))
    return found


def next_nodes(snapshot: Any) -> tuple[str, ...]:
    return tuple(getattr(snapshot, "next", ()) or ())


def derived_thread_id(first_user_text: str, user: str | None = None) -> str:
    """Stable dialog id for a client that resends the transcript."""
    material = f"{(user or '').strip()}\0{first_user_text.strip()}"
    return "c" + sha256(material.encode()).hexdigest()[:32]


def ensure_same_job(snapshot: Any, job_id: str | None) -> None:
    values = getattr(snapshot, "values", None) or {}
    existing = values.get("job_id")
    if existing and job_id and existing != job_id:
        raise JobMismatchError(existing)


def decide_input(snapshot: Any, text: str, job_id: str | None) -> Any:
    """Resume a pause, continue a crashed run, or start a turn.

    A crashed run (`next` set, no interrupt) is continued with ``None``.
    The new text is not treated as a new question.
    """
    ensure_same_job(snapshot, job_id)
    if interrupts_of(snapshot):
        return Command(resume=text)
    if next_nodes(snapshot):
        return None
    return new_turn_input(text, job_id)


def public_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for item in citations:
        row: dict[str, Any] = {
            "row_key": item.get("row_key"),
            "period_id": item.get("period_id"),
            "cell": item.get("cell"),
        }
        if item.get("value") not in (None, ""):
            row["value"] = item.get("value")
        status = item.get("value_status")
        if status:
            row["value_status"] = status
        public.append(row)
    return public


def render_completion(snapshot: Any, *, thread_id: str) -> dict[str, Any]:
    values = getattr(snapshot, "values", None) or {}
    awaiting = bool(interrupts_of(snapshot))
    content = values.get("user_question") if awaiting else (values.get("draft") or "")
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": AGENT_MODEL,
        "thread_id": thread_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content or ""},
                "finish_reason": "stop",
            }
        ],
        "awaiting_user": awaiting,
        "satisfactory": False if awaiting else bool(values.get("satisfactory")),
        "gaps": list(values.get("gaps") or []),
        "citations": public_citations(list(values.get("citations") or [])),
        "steps": list(values.get("steps") or []),
    }
