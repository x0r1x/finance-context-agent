"""One in-flight run per thread. The lock is not a graph field."""

from __future__ import annotations

from uuid import uuid4

import redis.asyncio as aioredis

LOCK_TTL_SECONDS = 900

_RELEASE = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""


class MemoryThreadLock:
    """Single-process lock for tests. Production uses Redis."""

    def __init__(self) -> None:
        self._held: dict[str, str] = {}

    async def acquire(self, thread_id: str) -> str | None:
        if thread_id in self._held:
            return None
        token = uuid4().hex
        self._held[thread_id] = token
        return token

    async def release(self, thread_id: str, token: str) -> None:
        if self._held.get(thread_id) == token:
            del self._held[thread_id]


class RedisThreadLock:
    def __init__(self, client: aioredis.Redis) -> None:
        self._redis = client

    async def acquire(self, thread_id: str) -> str | None:
        token = uuid4().hex
        ok = await self._redis.set(self._key(thread_id), token, nx=True, ex=LOCK_TTL_SECONDS)
        if not ok:
            return None
        return token

    async def release(self, thread_id: str, token: str) -> None:
        await self._redis.eval(_RELEASE, 1, self._key(thread_id), token)

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"lock:{thread_id}"
