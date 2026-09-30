"""HTTP surface. The workbook stays on the parser. The model stays on LLM_BASE_URL."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langgraph.errors import GraphInterrupt, GraphRecursionError
from pydantic import BaseModel, ConfigDict
from redis.exceptions import RedisError

from finance_context_agent.llm import ModelError
from finance_context_agent.parser import ParserError
from finance_context_agent.session import (
    JobMismatchError,
    decide_input,
    render_completion,
)
from finance_context_agent.turn import run_config

router = APIRouter()

_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = []
    stream: bool = False
    job_id: str | None = None
    thread_id: str | None = None


def error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


def last_user_text(messages: list[ChatMessage]) -> str:
    if not messages:
        raise ValueError("messages_required")
    last = messages[-1]
    if last.role != "user":
        raise ValueError("last_message_not_user")
    text = _content_text(last.content).strip()
    if not text:
        raise ValueError("user_message_required")
    return text


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    redis_ok = await _redis_ok(getattr(request.app.state, "redis", None))
    parser_ok = await _parser_ok(getattr(request.app.state, "parser", None))
    ready = redis_ok and parser_ok
    return JSONResponse(
        {"status": "ready" if ready else "unavailable", "redis": redis_ok, "parser": parser_ok},
        status_code=200 if ready else 503,
    )


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
    except (ModelError, ParserError, httpx.HTTPError, RedisError, GraphRecursionError):
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


async def _redis_ok(redis: Any) -> bool:
    if redis is None:
        return False
    try:
        return bool(await redis.ping())
    except Exception:
        return False


async def _parser_ok(parser: Any) -> bool:
    if parser is None:
        return False
    try:
        return bool(await parser.ready())
    except Exception:
        return False
