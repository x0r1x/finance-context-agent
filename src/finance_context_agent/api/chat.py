"""One chat turn. The lock is taken before the graph runs."""

from __future__ import annotations

import logging
import re
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langgraph.errors import GraphInterrupt, GraphRecursionError
from redis.exceptions import RedisError

from finance_context_agent.api.schemas import ChatRequest, last_user_text
from finance_context_agent.llm import ModelError
from finance_context_agent.parser import ParserError
from finance_context_agent.session import (
    JobMismatchError,
    decide_input,
    render_completion,
)
from finance_context_agent.turn import run_config

router = APIRouter()
logger = logging.getLogger("finance_context_agent.api")

_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest) -> JSONResponse:
    if body.stream:
        return error(400, "stream_unsupported")
    try:
        text = last_user_text(body.messages)
    except ValueError as exc:
        return error(400, str(exc))
    job_id = _optional_id(body.job_id or request.headers.get("x-job-id"), _JOB_ID, "bad_job_id")
    if isinstance(job_id, JSONResponse):
        return job_id
    thread_id = _optional_id(
        body.thread_id or request.headers.get("x-thread-id"),
        _THREAD_ID,
        "bad_thread_id",
    )
    if isinstance(thread_id, JSONResponse):
        return thread_id
    if thread_id is None:
        thread_id = str(uuid4())
    return await _run(request, thread_id=thread_id, job_id=job_id, text=text)


async def _run(request: Request, *, thread_id: str, job_id: str | None, text: str) -> JSONResponse:
    lock = request.app.state.lock
    token = await lock.acquire(thread_id)
    if token is None:
        return error(409, "thread_busy")
    config = run_config(thread_id, request.app.state.settings)
    graph = request.app.state.graph
    try:
        snapshot = await graph.aget_state(config)
        try:
            payload = decide_input(snapshot, text, job_id)
        except JobMismatchError:
            return error(409, "job_mismatch")
        try:
            await graph.ainvoke(payload, config, durability="sync")
        except GraphInterrupt:
            pass
        snapshot = await graph.aget_state(config)
    except (ModelError, ParserError, httpx.HTTPError, RedisError, GraphRecursionError) as exc:
        logger.warning("upstream failed", exc_info=exc)
        return error(503, "upstream_unavailable")
    finally:
        await lock.release(thread_id, token)
    return JSONResponse(render_completion(snapshot, thread_id=thread_id))


def _optional_id(
    value: str | None, pattern: re.Pattern[str], code: str
) -> str | None | JSONResponse:
    text = (value or "").strip()
    if not text:
        return None
    if pattern.fullmatch(text) is None:
        return error(400, code)
    return text
