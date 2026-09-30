import asyncio
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from finance_context_agent.app import create_app
from finance_context_agent.graph import build_graph
from finance_context_agent.lock import MemoryThreadLock
from finance_context_agent.parser import ParserError
from finance_context_agent.turn import run_config
from tests.fakes import (
    FakeParser,
    ScriptedModel,
    answer,
    catalog_row,
    cite,
    observation,
    page,
    plan,
    year_axes,
)

THREAD = "thread-1"
JOB = "job-1"


class GateModel:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete_json(self, *, role: str, system: str, user: str) -> dict:
        self.entered.set()
        await self.release.wait()
        if role == "plan":
            return plan(["DSCR"], [{"year": "2030"}])
        if role == "answer":
            return answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
        return {"gaps": []}


def _ready(parser: FakeParser, needle: str = "DSCR", key: str = "row-dscr") -> None:
    parser.axes = year_axes()
    parser.pages[(JOB, needle.casefold())] = page([catalog_row(key, needle)])
    parser.observations[(JOB, key)] = [observation(key, "2030", "1.25", "C10", label=needle)]


def _app(parser: FakeParser, model):
    graph = build_graph(parser, model, InMemorySaver())
    app = create_app(graph=graph, lock=MemoryThreadLock(), parser=parser, redis=None)
    return app, graph


def _payload(text: str, **extra) -> dict:
    body = {
        "model": "finance-context-agent",
        "messages": [{"role": "user", "content": text}],
        "thread_id": THREAD,
        "job_id": JOB,
    }
    body.update(extra)
    return body


@pytest.mark.asyncio
async def test_healthz_ignores_the_parser_and_readyz_checks_both() -> None:
    parser = FakeParser()
    parser.ready_ok = False
    app, _graph = _app(parser, ScriptedModel())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        health = await client.get("/healthz")
        ready = await client.get("/readyz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 503
    assert ready.json()["status"] == "unavailable"


@pytest.mark.asyncio
async def test_stream_and_non_user_message_are_rejected() -> None:
    app, _graph = _app(FakeParser(), ScriptedModel())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        streamed = await client.post("/v1/chat/completions", json=_payload("вопрос", stream=True))
        assistant = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "assistant", "content": "нет"}],
                "thread_id": THREAD,
                "job_id": JOB,
            },
        )
    assert streamed.status_code == 400
    assert streamed.json()["error"] == "stream_unsupported"
    assert assistant.status_code == 400
    assert assistant.json()["error"] == "last_message_not_user"


@pytest.mark.asyncio
async def test_missing_thread_id_is_minted() -> None:
    parser = FakeParser()
    _ready(parser)
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Какой DSCR в 2030?"}], "job_id": JOB},
        )
    body = response.json()
    assert response.status_code == 200
    UUID(body["thread_id"])
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["satisfactory"] is True
    assert body["awaiting_user"] is False
    assert body["citations"][0]["cell"] == "C10"


@pytest.mark.asyncio
async def test_missing_job_asks_and_a_different_job_conflicts() -> None:
    parser = FakeParser()
    parser.jobs = [{"job_id": JOB, "source_filename": "model.xlsx"}]
    _ready(parser)
    model = ScriptedModel()
    app, graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        asked = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Какой DSCR?"}], "thread_id": THREAD},
        )
        assert asked.status_code == 200
        assert asked.json()["awaiting_user"] is True
        assert "model.xlsx" in asked.json()["choices"][0]["message"]["content"]
        assert parser.observation_calls == []

        model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
        model.push(
            "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
        )
        model.push("critic", {"gaps": []})
        chosen = await client.post(
            "/v1/chat/completions",
            json=_payload("model.xlsx", job_id=JOB),
        )
        assert chosen.status_code == 200
        assert chosen.json()["satisfactory"] is True

        mismatch = await client.post(
            "/v1/chat/completions",
            json=_payload("ещё вопрос", job_id="job-2"),
        )
    assert mismatch.status_code == 409
    assert mismatch.json()["error"] == "job_mismatch"
    snap = await graph.aget_state(run_config(THREAD))
    assert snap.values["job_id"] == JOB


@pytest.mark.asyncio
async def test_two_dscr_rows_resume_through_the_route() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[(JOB, "dscr")] = page(
        [
            catalog_row("row-obs", "DSCR наблюдённый", concept="dscr.observed"),
            catalog_row("row-lim", "DSCR лимит", concept="dscr.limit", sheet="Limits"),
        ]
    )
    parser.pages[(JOB, "dscr наблюдённый")] = page(
        [catalog_row("row-obs", "DSCR наблюдённый", concept="dscr.observed")]
    )
    parser.observations[(JOB, "row-obs")] = [
        observation("row-obs", "2030", "1.25", "C10", label="DSCR наблюдённый")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        first = await client.post("/v1/chat/completions", json=_payload("Какой DSCR в 2030?"))
        assert first.status_code == 200
        body = first.json()
        assert body["awaiting_user"] is True
        assert "999" not in body["choices"][0]["message"]["content"]
        assert parser.observation_calls == []
        model.push("plan", plan(["DSCR наблюдённый"], [{"year": "2030"}]))
        model.push(
            "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-obs", "2030", "1.25", "C10")])
        )
        model.push("critic", {"gaps": []})
        second = await client.post(
            "/v1/chat/completions",
            json=_payload("Наблюдённый, не лимит"),
        )
    assert second.status_code == 200
    done = second.json()
    assert done["awaiting_user"] is False
    assert done["satisfactory"] is True
    assert done["citations"][0]["row_key"] == "row-obs"
    follow = [item for item in model.seen if item["role"] == "plan"][-1]["user"]
    assert "Наблюдённый" in follow
    assert "Какой DSCR в 2030?" in follow


@pytest.mark.asyncio
async def test_why_after_finish_is_a_new_turn() -> None:
    parser = FakeParser()
    _ready(parser)
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    model.push("plan", plan(["DSCR"], [{"year": "2030"}], "explain", "precedents"))
    model.push("answer", answer("1.25", [cite("row-dscr", "2030", "1.25", "C10")]))
    model.push("critic", {"gaps": []})
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        first = await client.post("/v1/chat/completions", json=_payload("Какой DSCR в 2030?"))
        assert first.json()["satisfactory"] is True
        second = await client.post("/v1/chat/completions", json=_payload("А почему?"))
    assert second.status_code == 200
    assert second.json()["awaiting_user"] is False
    assert second.json()["satisfactory"] is True
    follow = [item for item in model.seen if item["role"] == "plan"][-1]["user"]
    assert '"question": "А почему?"' in follow
    assert '"human_reply": ""' in follow
    assert "row-dscr" in follow


@pytest.mark.asyncio
async def test_thread_busy_while_the_lock_is_held() -> None:
    parser = FakeParser()
    _ready(parser)
    model = GateModel()
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_payload("Какой DSCR в 2030?"))
        )
        await model.entered.wait()
        second = await client.post("/v1/chat/completions", json=_payload("ещё"))
        assert second.status_code == 409
        assert second.json()["error"] == "thread_busy"
        model.release.set()
        finished = await first
    assert finished.status_code == 200
    assert finished.json()["satisfactory"] is True


@pytest.mark.asyncio
async def test_crashed_run_is_continued_instead_of_replacing_the_question() -> None:
    parser = FakeParser()
    _ready(parser)
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("answer", "boom")
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    app, graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        failed = await client.post("/v1/chat/completions", json=_payload("Какой DSCR в 2030?"))
        assert failed.status_code == 503
        assert failed.json()["error"] == "upstream_unavailable"
        continued = await client.post("/v1/chat/completions", json=_payload("другой вопрос"))
    assert continued.status_code == 200
    assert continued.json()["satisfactory"] is True
    snap = await graph.aget_state(run_config(THREAD))
    assert snap.values["question"] == "Какой DSCR в 2030?"


@pytest.mark.asyncio
async def test_parser_outage_is_unavailable_and_not_ready_is_an_answer() -> None:
    parser = FakeParser()
    parser.error = ParserError(502, "down")
    app, _graph = _app(parser, ScriptedModel())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        down = await client.post("/v1/chat/completions", json=_payload("Какой DSCR?"))
    assert down.status_code == 503

    parser = FakeParser()
    parser.error = ParserError(409, "report_not_ready")
    app, _graph = _app(parser, ScriptedModel())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        compiling = await client.post("/v1/chat/completions", json=_payload("Какой DSCR?"))
    assert compiling.status_code == 200
    body = compiling.json()
    assert body["satisfactory"] is False
    assert body["awaiting_user"] is False
    assert "собирается" in body["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_job_and_thread_headers_are_accepted() -> None:
    parser = FakeParser()
    _ready(parser)
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Job-Id": JOB, "X-Thread-Id": THREAD},
            json={"messages": [{"role": "user", "content": "Какой DSCR в 2030?"}]},
        )
    assert response.status_code == 200
    assert response.json()["thread_id"] == THREAD
    assert response.json()["satisfactory"] is True
