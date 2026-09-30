"""ASGI app. The graph and the Redis connection live for the process."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from finance_context_agent.api import router
from finance_context_agent.graph import build_graph
from finance_context_agent.llm import OpenAIChat
from finance_context_agent.lock import RedisThreadLock
from finance_context_agent.parser import ParserClient
from finance_context_agent.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    parser = ParserClient(settings.resolved_parser_base_url(), timeout=settings.parser_timeout_sec)
    model = OpenAIChat(
        settings.resolved_llm_base_url(),
        settings.llm_api_key,
        settings.llm_model,
        timeout=settings.llm_timeout_sec,
        temperature=settings.llm_temperature,
    )
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    async with AsyncRedisSaver.from_conn_string(
        settings.redis_url, ttl=settings.saver_ttl()
    ) as checkpointer:
        await checkpointer.asetup()
        app.state.graph = build_graph(parser, model, checkpointer, settings)
        app.state.redis = redis
        app.state.lock = RedisThreadLock(redis, ttl_seconds=settings.lock_ttl_sec)
        app.state.parser = parser
        app.state.model = model
        try:
            yield
        finally:
            await parser.aclose()
            await model.aclose()
            await redis.aclose()


def create_app(
    settings: Settings | None = None,
    *,
    graph: Any = None,
    lock: Any = None,
    parser: Any = None,
    redis: Any = None,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan if graph is None else None)
    app.state.settings = settings or Settings()
    if graph is not None:
        app.state.graph = graph
        app.state.lock = lock
        app.state.parser = parser
        app.state.redis = redis
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "finance_context_agent.app:app",
        host=settings.host,
        port=settings.port,
    )
