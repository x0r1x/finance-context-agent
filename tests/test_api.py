import asyncio
import json
import logging
import re

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from finance_context_agent.app import create_app
from finance_context_agent.graph import build_graph
from finance_context_agent.lock import MemoryThreadLock
from finance_context_agent.parser import ParserError
from finance_context_agent.session import derived_thread_id
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
_DERIVED = re.compile(r"^c[0-9a-f]{32}$")


class GateModel:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete_json(
        self, *, role: str, system: str, user: str, options: list[str] | None = None
    ) -> dict:
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
    parser = FakeParser()
    parser.jobs = [{"job_id": JOB, "source_filename": "model.xlsx"}]
    model = ScriptedModel()
    model.push("about", {"acts": ["row"]})
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        models = await client.get("/v1/models")
        streamed = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "Какой DSCR?"}],
                "thread_id": THREAD,
            },
        )
        assistant = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "assistant", "content": "нет"}],
                "thread_id": THREAD,
                "job_id": JOB,
            },
        )
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "finance-context-agent"
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in streamed.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert streamed.text.rstrip().endswith("data: [DONE]")
    content = frames[0]["choices"][0]["delta"]["content"]
    assert "Какую книгу открыть?" in content
    assert "model.xlsx" in content
    assert "\n" not in streamed.text.split("data: ")[1].split("\n", 1)[0]
    assert frames[1]["choices"][0]["finish_reason"] == "stop"
    assert assistant.status_code == 400
    assert assistant.json()["error"] == "last_message_not_user"


@pytest.mark.asyncio
async def test_missing_thread_id_is_minted() -> None:
    parser = FakeParser()
    _ready(parser)
    model = ScriptedModel()
    question = "Какой DSCR в 2030?"
    for _ in range(2):
        model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
        model.push(
            "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
        )
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": question}], "job_id": JOB},
        )
        again = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": question}], "job_id": JOB},
        )
        other = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "job_id": JOB,
                "user": "alice",
            },
        )
        model.push("about", {"acts": ["chitchat"]})
        model.push("about", {"acts": ["row"]})
        model.push("about", {"acts": ["chitchat"]})
        hello = "привет"
        opened_hello = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": hello}]},
        )
        hello_text = opened_hello.json()["choices"][0]["message"]["content"]
        paused_hello = await client.post(
            "/v1/chat/completions",
            json={
                "job_id": JOB,
                "messages": [
                    {"role": "user", "content": hello},
                    {"role": "assistant", "content": hello_text},
                    {"role": "user", "content": "Какой ZZZ?"},
                ],
            },
        )
        fresh_hello = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": hello}]},
        )
    body = response.json()
    assert response.status_code == 200
    assert _DERIVED.fullmatch(body["thread_id"])
    assert body["thread_id"] == derived_thread_id(question)
    assert again.json()["thread_id"] == body["thread_id"]
    assert other.json()["thread_id"] == derived_thread_id(question, "alice")
    assert other.json()["thread_id"] != body["thread_id"]
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["satisfactory"] is True
    assert body["awaiting_user"] is False
    assert body["citations"][0]["cell"] == "C10"
    assert opened_hello.status_code == 200
    assert opened_hello.json()["thread_id"] == derived_thread_id(hello)
    assert opened_hello.json()["awaiting_user"] is False
    assert "finance-context-agent" in hello_text
    assert paused_hello.json()["thread_id"] == derived_thread_id(hello)
    assert paused_hello.json()["awaiting_user"] is True
    assert "Такой строки нет" in paused_hello.json()["choices"][0]["message"]["content"]
    fresh_body = fresh_hello.json()
    assert fresh_hello.status_code == 200
    assert fresh_body["thread_id"] == derived_thread_id(hello)
    assert fresh_body["awaiting_user"] is False
    assert "Такой строки нет" not in fresh_body["choices"][0]["message"]["content"]
    assert "finance-context-agent" in fresh_body["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_missing_job_asks_and_a_different_job_conflicts() -> None:
    parser = FakeParser()
    parser.jobs = [{"job_id": JOB, "source_filename": "model.xlsx"}]
    _ready(parser)
    model = ScriptedModel()
    model.push("about", {"acts": ["row"]})
    model.push("about", {"acts": ["row"]})
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
        opened = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Какой DSCR?"}]},
        )
        pause = opened.json()["choices"][0]["message"]["content"]
        seen = len(model.seen)
        repeated = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Какой DSCR?"}]},
        )
        assert len(model.seen) == seen
        model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
        model.push(
            "answer",
            answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")]),
        )
        followed = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Какой DSCR?"},
                    {"role": "assistant", "content": pause},
                    {"role": "user", "content": "model.xlsx"},
                ]
            },
        )
    assert mismatch.status_code == 409
    assert mismatch.json()["error"] == "job_mismatch"
    snap = await graph.aget_state(run_config(THREAD))
    assert snap.values["job_id"] == JOB
    assert opened.status_code == 200
    derived = opened.json()["thread_id"]
    assert opened.json()["awaiting_user"] is True
    assert "model.xlsx" in pause
    assert repeated.json()["thread_id"] == derived
    assert repeated.json()["choices"][0]["message"]["content"] == pause
    assert followed.status_code == 200
    assert followed.json()["thread_id"] == derived
    assert followed.json()["satisfactory"] is True
    derived_state = await graph.aget_state(run_config(derived))
    assert derived_state.values["job_id"] == JOB


@pytest.mark.asyncio
async def test_thread_busy_while_the_lock_is_held() -> None:
    parser = FakeParser()
    _ready(parser)
    model = GateModel()
    app, _graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json=_payload("Какой?")))
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
    app, graph = _app(parser, model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
        failed = await client.post("/v1/chat/completions", json=_payload("Какой?"))
        assert failed.status_code == 503
        assert failed.json()["error"] == "upstream_unavailable"
        continued = await client.post("/v1/chat/completions", json=_payload("другой вопрос"))
    assert continued.status_code == 200
    assert continued.json()["satisfactory"] is True
    snap = await graph.aget_state(run_config(THREAD))
    assert snap.values["question"] == "Какой?"


@pytest.mark.asyncio
async def test_model_error_is_logged_and_hidden(caplog: pytest.LogCaptureFixture) -> None:
    parser = FakeParser()
    _ready(parser)
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("answer", "boom")
    app, _graph = _app(parser, model)
    with caplog.at_level(logging.WARNING, logger="finance_context_agent.api"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://agent") as client:
            failed = await client.post("/v1/chat/completions", json=_payload("Какой?"))
    assert failed.status_code == 503
    assert failed.json() == {"error": "upstream_unavailable"}
    assert "boom" in caplog.text
    assert "boom" not in failed.text


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
