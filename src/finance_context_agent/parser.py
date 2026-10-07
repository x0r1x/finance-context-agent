"""HTTP client for finance-context-builder slice routes.

ETag is a property of the workbook files, not of the query string. A cached
body may be reused only for the same URL that produced it.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import httpx

from finance_context_agent.settings import field_default


class ParserError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(f"{status} {code}")


class ParserClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        seconds = field_default("parser_timeout_sec") if timeout is None else timeout
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(float(seconds)))
        self._owns_client = client is None
        self._bodies: dict[str, Any] = {}
        self._etags: dict[str, str] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ready(self) -> bool:
        response = await self._client.get(f"{self._base}/readyz")
        return response.status_code == 200

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = []
        if status:
            params.append(("status", status))
        if q:
            params.append(("q", q))
        body, _etag = await self._get("/v1/context-jobs", params)
        if not isinstance(body, list):
            raise ParserError(502, "bad_jobs")
        return body

    async def get_summary(self, job_id: str) -> tuple[dict[str, Any], str]:
        body, etag = await self._get(f"/v1/context-jobs/{job_id}/summary", [])
        return body, etag

    async def get_context(self, job_id: str) -> dict[str, Any]:
        """Whole context document. Too large to keep in the slice cache."""
        url = f"{self._base}/v1/context-jobs/{job_id}/context.json"
        response = await self._client.get(url)
        self._raise_for_status(response)
        body = response.json()
        if not isinstance(body, dict):
            raise ParserError(502, "bad_context")
        return body

    async def head_catalog_etag(self, job_id: str) -> str:
        url = self._full(f"/v1/context-jobs/{job_id}/catalog", [("limit", "1")])
        response = await self._client.request("HEAD", url)
        self._raise_for_status(response)
        etag = response.headers.get("etag") or ""
        return etag

    async def get_axes(self, job_id: str) -> tuple[dict[str, Any], str]:
        """Axes and total. The single returned row is not a search result."""
        body, etag = await self._get(
            f"/v1/context-jobs/{job_id}/catalog",
            [("limit", "1")],
        )
        return body, etag

    async def search_rows(self, job_id: str, q: str, *, limit: int = 8) -> dict[str, Any]:
        body, _etag = await self._get(
            f"/v1/context-jobs/{job_id}/catalog",
            [("q", q), ("limit", str(limit))],
        )
        return body

    async def get_observations(
        self,
        job_id: str,
        *,
        row_key: str,
        period_ids: list[str] | None = None,
        precedent_depth: int = 0,
        limit: int = 24,
    ) -> dict[str, Any]:
        params: list[tuple[str, str]] = [
            ("row_key", row_key),
            ("precedent_depth", str(precedent_depth)),
            ("limit", str(limit)),
        ]
        for period_id in period_ids or []:
            params.append(("period_id", period_id))
        body, _etag = await self._get(f"/v1/context-jobs/{job_id}/observations", params)
        return body

    async def trace_dependents(self, job_id: str, origin: str, *, depth: int = 2) -> dict[str, Any]:
        body, _etag = await self._get(
            f"/v1/context-jobs/{job_id}/graph/trace",
            [("from", origin), ("direction", "dependents"), ("depth", str(depth))],
        )
        return body

    def remembers(self, path: str, params: list[tuple[str, str]]) -> bool:
        return self._full(path, params) in self._bodies

    async def _get(self, path: str, params: list[tuple[str, str]]) -> tuple[Any, str]:
        url = self._full(path, params)
        headers: dict[str, str] = {}
        cached_etag = self._etags.get(url)
        if url in self._bodies and cached_etag:
            headers["If-None-Match"] = cached_etag
        response = await self._client.get(url, headers=headers)
        if response.status_code == 304:
            return self._bodies[url], cached_etag or ""
        self._raise_for_status(response)
        body = response.json()
        etag = response.headers.get("etag") or ""
        self._bodies[url] = body
        if etag:
            self._etags[url] = etag
        return body, etag

    def _full(self, path: str, params: list[tuple[str, str]]) -> str:
        query = urlencode(params)
        if not query:
            return f"{self._base}{path}"
        return f"{self._base}{path}?{query}"

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        code = "parser_error"
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            code = payload["error"]
        raise ParserError(response.status_code, code)
