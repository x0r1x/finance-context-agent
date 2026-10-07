"""Liveness and readiness. Readiness names Redis and the parser only."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


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
