"""One chat turn. The lock is taken before the graph runs."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.errors import GraphInterrupt, GraphRecursionError
from redis.exceptions import RedisError

from finance_context_agent.api.schemas import (
    ChatRequest,
    first_user_text,
    history_has_assistant,
    last_user_text,
)
from finance_context_agent.llm import ModelError
from finance_context_agent.parser import ParserError
from finance_context_agent.session import (
    JobMismatchError,
    decide_input,
    derived_thread_id,
    ensure_same_job,
    interrupts_of,
    render_completion,
)
from finance_context_agent.turn import run_config

router = APIRouter()
logger = logging.getLogger("finance_context_agent.api")

_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request, body: ChatRequest) -> JSONResponse | StreamingResponse:
    try:
        text = last_user_text(body.messages)
    except ValueError as exc:
        return error(400, str(exc))
    job_id = _optional_id(body.job_id or request.headers.get("x-job-id"), _JOB_ID, "bad_job_id")
    if isinstance(job_id, JSONResponse):
        return job_id
    explicit = _optional_id(
        body.thread_id or request.headers.get("x-thread-id"),
        _THREAD_ID,
        "bad_thread_id",
    )
    if isinstance(explicit, JSONResponse):
        return explicit
    derived = explicit is None
    thread_id = explicit or derived_thread_id(first_user_text(body.messages), body.user)
    return await _run(
        request,
        thread_id=thread_id,
        job_id=job_id,
        text=text,
        derived=derived,
        follow_up=history_has_assistant(body.messages),
        stream=body.stream,
    )


async def _run(
    request: Request,
    *,
    thread_id: str,
    job_id: str | None,
    text: str,
    derived: bool,
    follow_up: bool,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    lock = request.app.state.lock
    token = await lock.acquire(thread_id)
    if token is None:
        return error(409, "thread_busy")
    config = run_config(thread_id, request.app.state.settings)
    graph = request.app.state.graph
    completion: dict[str, Any] | None = None
    try:
        snapshot = await graph.aget_state(config)
        stored_question = str((getattr(snapshot, "values", None) or {}).get("question") or "")
        repeat_pause = (
            derived
            and not follow_up
            and bool(interrupts_of(snapshot))
            and text.strip() == stored_question.strip()
        )
        try:
            if repeat_pause:
                ensure_same_job(snapshot, job_id)
            else:
                payload = decide_input(snapshot, text, job_id)
                try:
                    await graph.ainvoke(payload, config, durability="sync")
                except GraphInterrupt:
                    pass
                snapshot = await graph.aget_state(config)
        except JobMismatchError:
            return error(409, "job_mismatch")
        completion = render_completion(snapshot, thread_id=thread_id)
    except (ModelError, ParserError, httpx.HTTPError, RedisError, GraphRecursionError) as exc:
        logger.warning("upstream failed", exc_info=exc)
        return error(503, "upstream_unavailable")
    finally:
        await lock.release(thread_id, token)
    assert completion is not None
    if stream:
        return _event_stream(completion)
    return JSONResponse(completion)


def _event_stream(completion: dict[str, Any]) -> StreamingResponse:
    message = completion["choices"][0]["message"]["content"]
    base = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    }

    async def chunks() -> AsyncIterator[str]:
        yield _frame(base, {"role": "assistant", "content": message}, None)
        yield _frame(base, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def _frame(base: dict[str, Any], delta: dict[str, Any], finish: str | None) -> str:
    payload = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _optional_id(
    value: str | None, pattern: re.Pattern[str], code: str
) -> str | None | JSONResponse:
    text = (value or "").strip()
    if not text:
        return None
    if pattern.fullmatch(text) is None:
        return error(400, code)
    return text
