"""Pause across two graph objects on a real Redis 8. fakeredis does not apply."""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest
import redis.asyncio as aioredis
from langgraph.types import Command

from finance_context_agent.graph import build_graph
from finance_context_agent.lock import RedisThreadLock
from finance_context_agent.settings import Settings
from finance_context_agent.turn import new_turn_input, run_config
from tests.fakes import (
    FakeParser,
    ScriptedModel,
    catalog_row,
    observation,
    page,
    year_axes,
)

pytestmark = pytest.mark.asyncio


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, timeout=180)


@pytest.fixture
async def redis_url():
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    name = "fca-redis-test"
    _docker("docker", "rm", "-f", name)
    started = _docker(
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        "127.0.0.1:6399:6379",
        "redis:8",
        "redis-server",
        "--appendonly",
        "yes",
        "--appendfsync",
        "everysec",
        "--requirepass",
        "testpass",
        "--maxmemory",
        "64mb",
        "--maxmemory-policy",
        "noeviction",
    )
    if started.returncode != 0:
        pytest.skip(started.stderr.strip() or "redis:8 did not start")
    url = "redis://:testpass@127.0.0.1:6399/0"
    client = aioredis.from_url(url, decode_responses=True)
    deadline = time.monotonic() + 30
    try:
        while True:
            try:
                if await client.ping():
                    break
            except Exception:
                if time.monotonic() > deadline:
                    pytest.skip("redis did not answer ping")
                await _asleep()
            else:
                break
        yield url
    finally:
        await client.aclose()
        _docker("docker", "rm", "-f", name)


async def _asleep() -> None:
    import asyncio

    await asyncio.sleep(0.2)


async def test_live_redis_pauses_and_the_lock_is_owned(redis_url: str) -> None:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "dscr")] = page(
        [
            catalog_row("row-obs", "DSCR наблюдённый", concept="dscr.observed"),
            catalog_row("row-lim", "DSCR лимит", concept="dscr.limit", sheet="Limits"),
        ]
    )
    parser.observations[("job-1", "row-obs")] = [
        observation("row-obs", "2030", "1.25", "C10", label="DSCR наблюдённый")
    ]
    model = ScriptedModel()
    config = run_config("thread-redis")
    ttl = Settings(_env_file=None).saver_ttl()

    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:
        await saver.asetup()
        graph = build_graph(parser, model, saver)
        await graph.ainvoke(
            new_turn_input("Какой DSCR в 2030?", "job-1"), config, durability="sync"
        )
        paused = await graph.aget_state(config)
        assert paused.next

    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:
        await saver.asetup()
        graph = build_graph(parser, model, saver)
        await graph.ainvoke(Command(resume="DSCR наблюдённый"), config, durability="sync")
        done = await graph.aget_state(config)
    assert done.values["satisfactory"] is True
    assert done.values["citations"][0]["row_key"] == "row-obs"

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        lock = RedisThreadLock(client)
        token = await lock.acquire("thread-redis")
        assert token
        assert await lock.acquire("thread-redis") is None
        await lock.release("thread-redis", "other-token")
        assert await lock.acquire("thread-redis") is None
        await lock.release("thread-redis", token)
        again = await lock.acquire("thread-redis")
        assert again
        await lock.release("thread-redis", again)
    finally:
        await client.aclose()
