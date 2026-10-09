"""Dialogue-act embeddings. One POST, then cosine against cached anchors."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

import httpx

from finance_context_agent.clients.llm import ModelError
from finance_context_agent.questions import ANCHORS, act_text, class_scores
from finance_context_agent.settings import field_default

logger = logging.getLogger(__name__)

_BATCH = 16


def embeddings_url(base_url: str) -> str:
    parts = urlsplit(base_url.strip())
    path = parts.path.rstrip("/")
    if path == "":
        path = "/v1"
    if not path.endswith("/embeddings"):
        path = f"{path}/embeddings"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = embeddings_url(base_url)
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        seconds = field_default("llm_timeout_sec") if timeout is None else timeout
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(float(seconds)))
        self._owns_client = client is None
        self._anchors: dict[str, list[list[float]]] | None = None
        self._anchor_model = ""

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def score(self, text: str) -> dict[str, float]:
        anchors = await self._anchor_vectors()
        vectors = await self._embed([text])
        if not vectors or not vectors[0]:
            raise ModelError("embed empty")
        return class_scores(vectors[0], anchors)

    async def _anchor_vectors(self) -> dict[str, list[list[float]]]:
        if self._anchors is not None and self._anchor_model == self._model:
            return self._anchors
        texts = [act_text(book, last, row, reply) for _act, book, last, row, reply in ANCHORS]
        vectors = await self._embed(texts)
        if len(vectors) != len(ANCHORS):
            raise ModelError("embed anchors")
        grouped: dict[str, list[list[float]]] = {}
        for anchor, vector in zip(ANCHORS, vectors, strict=True):
            if not vector:
                raise ModelError("embed empty")
            grouped.setdefault(anchor[0], []).append(vector)
        self._anchors = grouped
        self._anchor_model = self._model
        logger.info("embed anchors %s %s", self._model, len(ANCHORS))
        return grouped

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            chunk = texts[start : start + _BATCH]
            payload = {"model": self._model, "input": chunk}
            try:
                response = await self._client.post(
                    self._url, json=payload, headers=self._headers
                )
            except httpx.HTTPError as exc:
                raise ModelError(str(exc)) from exc
            if response.status_code >= 400:
                raise ModelError(
                    str(response.status_code),
                    status=response.status_code,
                    body=response.text[:300],
                )
            body = response.json()
            rows = body.get("data") if isinstance(body, dict) else None
            if not isinstance(rows, list) or len(rows) != len(chunk):
                raise ModelError("embed shape")
            ordered = sorted(rows, key=lambda item: int(item.get("index", 0)))
            for item in ordered:
                embedding = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(embedding, list):
                    raise ModelError("embed shape")
                vectors.append([float(value) for value in embedding])
        return vectors
