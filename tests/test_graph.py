import asyncio
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from finance_context_agent.graph import _choice_question, build_graph
from finance_context_agent.llm import ModelError
from finance_context_agent.parser import ParserError
from finance_context_agent.session import interrupts_of
from finance_context_agent.turn import new_turn_input, run_config
from tests.fakes import (
    ByQuestion,
    FakeParser,
    ScriptedModel,
    answer,
    catalog_row,
    cite,
    observation,
    page,
    plan,
    two_first_years,
    year_axes,
)


def _graph(parser: FakeParser, model) -> object:
    return build_graph(parser, model, InMemorySaver())


async def _run(graph, question: str, thread: str = "thread-1", job: str = "job-1"):
    await graph.ainvoke(new_turn_input(question, job), run_config(thread), durability="sync")


def _bind(parser: FakeParser, needle: str, rows: list[dict], job: str = "job-1") -> None:
    parser.axes = year_axes()
    parser.pages[(job, needle.casefold())] = page(rows)


def _seen(model: ScriptedModel, role: str) -> list[dict[str, str]]:
    return [item for item in model.seen if item["role"] == role]


def _slim(values: dict) -> None:
    blob = json.dumps(values, ensure_ascii=False)
    assert "CATALOG_PAGE" not in blob
    assert "DISCARD_ROW" not in blob
    assert "page_marker" not in blob
    assert values["summary"]["marker"] == "passport"
    assert values["axes"][0]["periods"]


def test_full_catalog_page_does_not_say_the_list_is_short() -> None:
    text = _choice_question(
        [("DSCR", page([catalog_row("a", "Observed"), catalog_row("b", "Limit")]))]
    )
    assert "Какую строку" in text
    assert "из " not in text


@pytest.mark.asyncio
async def test_short_catalog_page_says_how_many_labels_are_shown() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    rows = [catalog_row(f"row-{index}", f"Debt {index}") for index in range(8)]
    parser.pages[("job-1", "debt")] = page(rows, total=26)
    model = ScriptedModel()
    model.push("plan", plan(["Debt"]))
    graph = _graph(parser, model)
    await _run(graph, "Какой?")
    question = (await graph.aget_state(run_config("thread-1"))).values["user_question"]
    assert "Какую строку" in question
    assert "8 из 26" in question


@pytest.mark.asyncio
async def test_zero_hit_needle_does_not_hide_one_row() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
    row = catalog_row("P&L|13|P&L!r2", "EBITDA", concept="pnl.ebitda")
    parser.pages[("job-1", "ebitda")] = page([row])
    parser.observations[("job-1", "P&L|13|P&L!r2")] = [
        observation(
            "P&L|13|P&L!r2",
            "Y1",
            "0",
            "C10",
            status="zero_explicit",
            label="EBITDA",
        )
    ]
    model = ScriptedModel()
    model.push(
        "plan",
        plan(["EBITDA", "Y1", "financial_data"], [{"year": "Y1"}], "explain", "none"),
    )
    model.push(
        "answer",
        answer(
            "EBITDA в Y1 равен 0.",
            [cite("P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit")],
        ),
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    queries = [call["q"] for call in parser.catalog_calls]
    assert "EBITDA" in queries
    assert "Y1" not in queries
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "0"
    assert snap.values["citations"][0]["value_status"] == "zero_explicit"
    assert "999" not in snap.values["draft"]


@pytest.mark.asyncio
async def test_compose_keeps_a_missing_metric() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "ebitda")] = page([catalog_row("row-ebitda", "EBITDA")])
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Сравни EBITDA и DSCR")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "Какую строку" in snap.values["user_question"]
    assert "DSCR" in snap.values["user_question"]
    assert parser.observation_calls == []
    assert {call["q"] for call in parser.catalog_calls} == {"EBITDA", "DSCR"}


@pytest.mark.asyncio
async def test_currency_nag_does_not_hide_an_explicit_zero() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
    row = catalog_row("P&L|13|P&L!r2", "EBITDA", concept="pnl.ebitda")
    parser.pages[("job-1", "ebitda")] = page([row])
    parser.observations[("job-1", "P&L|13|P&L!r2")] = [
        observation(
            "P&L|13|P&L!r2",
            "Y1",
            "0",
            "C10",
            status="zero_explicit",
            scale_factor=1000,
            scale="k",
            label="EBITDA",
        )
    ]
    model = ScriptedModel()
    model.push("plan", plan(["Какой EBITDA"], [], "lookup", "none"))
    model.push(
        "answer",
        answer(
            "EBITDA в Y1 составляет 0 GBP.",
            [cite("P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit")],
        ),
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["gaps"] == []
    assert snap.values["citations"][0]["row_key"] == "P&L|13|P&L!r2"
    assert snap.values["citations"][0]["period_id"] == "Y1"
    assert snap.values["citations"][0]["value"] == "0"
    assert snap.values["citations"][0]["value_status"] == "zero_explicit"
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_question_prefix_still_finds_the_named_period() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
    row = catalog_row("P&L|13|P&L!r2", "EBITDA", concept="pnl.ebitda")
    parser.pages[("job-1", "ebitda")] = page([row])
    parser.observations[("job-1", "P&L|13|P&L!r2")] = [
        observation("P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit", label="EBITDA")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["Какой EBITDA"], [], "lookup", "none"))
    model.push(
        "answer",
        answer(
            "EBITDA в Y1 равен 0.",
            [cite("P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit")],
        ),
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    queries = [call["q"] for call in parser.catalog_calls]
    assert queries == ["EBITDA"]
    assert "Y1" not in queries
    assert parser.observation_calls[0]["period_ids"] == ["Y1"]
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "0"
    assert snap.values["citations"][0]["value_status"] == "zero_explicit"


@pytest.mark.asyncio
async def test_russian_debt_word_is_not_rewritten_to_debt() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    rows = [catalog_row(f"row-{index}", f"Debt {index}") for index in range(8)]
    parser.pages[("job-1", "debt")] = page(rows, total=26)
    model = ScriptedModel()
    model.push("plan", plan(["Какой долг"]))
    model.push("plan", plan(["Какой долг"]))
    graph = _graph(parser, model)
    await _run(graph, "Какой долг?")
    snap = await graph.aget_state(run_config("thread-1"))
    queries = [call["q"] for call in parser.catalog_calls]
    assert queries == ["долг", "долг"]
    assert "Debt" not in queries
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert parser.observation_calls == []


@pytest.mark.asyncio
async def test_period_key_leaves_one_catalog_query() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
    row = catalog_row("P&L|13|P&L!r2", "EBITDA", concept="pnl.ebitda")
    parser.pages[("job-1", "ebitda")] = page([row])
    parser.observations[("job-1", "P&L|13|P&L!r2")] = [
        observation("P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit", label="EBITDA")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["EBITDA Y1"], []))
    model.push(
        "answer",
        answer(
            "EBITDA в Y1 равен 0.",
            [cite("P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit")],
        ),
    )
    graph = _graph(parser, model)
    await _run(graph, "EBITDA Y1")
    snap = await graph.aget_state(run_config("thread-1"))
    assert [call["q"] for call in parser.catalog_calls] == ["EBITDA"]
    assert parser.observation_calls[0]["period_ids"] == ["Y1"]
    assert snap.values["citations"][0]["value"] == "0"


@pytest.mark.asyncio
async def test_a_hit_phrase_keeps_words_like_in() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "cash in bank")] = page(
        [catalog_row("row-cash", "Cash in Bank", concept="cash")]
    )
    parser.pages[("job-1", "cash")] = page([catalog_row("row-other", "Cash")], total=4)
    parser.observations[("job-1", "row-cash")] = [
        observation("row-cash", "Y1", "5", "C1", label="Cash in Bank")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["Cash in Bank"], [{"period_key": "Y1"}]))
    model.push("answer", answer("5", [cite("row-cash", "Y1", "5", "C1")]))
    graph = _graph(parser, model)
    await _run(graph, "Cash in Bank")
    assert [call["q"] for call in parser.catalog_calls] == ["Cash in Bank"]


@pytest.mark.asyncio
async def test_one_value_requests_one_row_and_named_periods() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    assert parser.observation_calls == [
        {
            "job_id": "job-1",
            "row_key": "row-dscr",
            "period_ids": ["2030"],
            "precedent_depth": 0,
            "limit": 48,
        }
    ]
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.next == ()
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["cell"] == "C10"
    assert snap.values["citations"][0]["value"] == "1.25"
    _slim(snap.values)
    assert _seen(model, "answer") == []
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_two_years_of_one_row_are_one_observation_call() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "2030", "1.25", "C10"),
        observation("row-dscr", "Y1", "1.10", "C2"),
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}, {"period_key": "Y1"}]))
    model.push("answer", answer("В 2030 DSCR 1.25.", [cite("row-dscr", "2030", "1.25", "C10")]))
    graph = _graph(parser, model)
    await _run(graph, "Сравни DSCR в 2030 и в Y1")
    assert len(parser.observation_calls) == 1
    assert parser.observation_calls[0]["period_ids"] == ["2030", "Y1"]
    assert parser.observation_calls[0]["row_key"] == "row-dscr"


@pytest.mark.asyncio
async def test_number_outside_observations_causes_another_step() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("answer", answer("Это 99.", [cite("row-dscr", "2030", "99", "C10")]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой?")
    assert len(_seen(model, "answer")) == 2
    assert all(item["role"] != "critic" for item in model.seen)
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "1.25"


@pytest.mark.asyncio
async def test_why_at_depth_zero_retries_at_depth_two_without_asking() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    stored = observation(
        "row-dscr",
        "2030",
        "1.25",
        "C10",
        label="DSCR",
        precedents=[
            {"cell": "H10", "label": "CFADS", "value": "15", "depth": 1},
            {"cell": "Z9", "label": "глубокий", "value": "99", "depth": 2},
        ],
    )
    stored["formula"]["text"] = "=H10/I10"
    parser.observations[("job-1", "row-dscr")] = [stored]
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Как считается DSCR в 2030?")
    assert parser.catalog_calls[0]["q"] == "DSCR"
    assert [call["precedent_depth"] for call in parser.observation_calls] == [2]
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values.get("awaiting") in ("", None)
    assert snap.values["satisfactory"] is True
    draft = snap.values["draft"]
    assert "=H10/I10" in draft
    assert "H10" in draft
    assert "15" in draft
    assert "Z9" not in draft
    assert "99" not in draft


@pytest.mark.asyncio
async def test_repeated_gap_is_not_satisfactory() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("answer", answer("Это 99.", [cite("row-dscr", "2030", "99", "C10")]))
    model.push("answer", answer("Это 99.", [cite("row-dscr", "2030", "99", "C10")]))
    graph = _graph(parser, model)
    await _run(graph, "Какой?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is False
    assert "Подтверждённого числа" in snap.values["draft"]
    assert len(_seen(model, "answer")) == 2
    assert any(item.startswith("number:") for item in snap.values["gaps"])


@pytest.mark.asyncio
async def test_empty_cache_finishes_without_a_zero() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "Y1", "", "C10", status="empty")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"period_key": "Y1"}]))
    model.push(
        "answer", answer("Ячейка empty.", [cite("row-dscr", "Y1", None, "C10", status="empty")])
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert "0" not in snap.values["draft"]
    assert snap.values["citations"][0]["value_status"] == "empty"


@pytest.mark.asyncio
async def test_two_labels_ask_and_do_not_fetch_observations() -> None:
    parser = FakeParser()
    _bind(
        parser,
        "DSCR",
        [
            catalog_row("row-obs", "DSCR наблюдённый", concept="dscr.observed"),
            catalog_row("row-lim", "DSCR лимит", concept="dscr.limit", sheet="Limits"),
        ],
    )
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert parser.observation_calls == []
    question = snap.values["user_question"]
    assert "Какую строку" in question
    assert "999" not in question
    assert "1.25" not in question
    _slim(snap.values)


@pytest.mark.asyncio
async def test_two_misses_ask_that_the_row_is_missing() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    model = ScriptedModel()
    model.push("plan", plan(["ZZZ"]))
    model.push("plan", plan(["ZZZ"]))
    graph = _graph(parser, model)
    await _run(graph, "ZZZ?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert len(parser.catalog_calls) == 2
    assert parser.observation_calls == []


@pytest.mark.asyncio
async def test_two_metrics_make_two_observation_calls() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "выручка")] = page([catalog_row("row-rev", "Выручка", concept="rev")])
    parser.pages[("job-1", "ebitda")] = page(
        [catalog_row("row-ebitda", "EBITDA", concept="ebitda")]
    )
    parser.observations[("job-1", "row-rev")] = [
        observation("row-rev", "2030", "10", "B2", label="Выручка")
    ]
    parser.observations[("job-1", "row-ebitda")] = [
        observation("row-ebitda", "2030", "4", "B3", label="EBITDA")
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Сравни Выручка и EBITDA в 2030")
    assert len(parser.observation_calls) == 2
    assert {call["row_key"] for call in parser.observation_calls} == {"row-rev", "row-ebitda"}
    assert sum(call["limit"] for call in parser.observation_calls) <= 48
    for call in parser.observation_calls:
        assert "concept_id" not in call
        assert "q" not in call
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True


@pytest.mark.asyncio
async def test_one_phase_year_resolves_without_a_question() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "Y1", "1.1", "C2")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"phase_year": 1}]))
    model.push("answer", answer("В первом году 1.1.", [cite("row-dscr", "Y1", "1.1", "C2")]))
    graph = _graph(parser, model)
    await _run(graph, "DSCR в первый операционный год")
    assert parser.observation_calls[0]["period_ids"] == ["Y1"]
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True


@pytest.mark.asyncio
async def test_two_phase_years_ask_without_inventing_a_period() -> None:
    parser = FakeParser()
    parser.axes = two_first_years()
    parser.pages[("job-1", "dscr")] = page([catalog_row("row-dscr", "DSCR")])
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"phase_year": 1}]))
    graph = _graph(parser, model)
    await _run(graph, "DSCR в первый операционный год")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert parser.observation_calls == []
    assert "нескольким" in snap.values["user_question"]
    assert "Y1" not in snap.values["user_question"]


@pytest.mark.asyncio
async def test_two_dscr_rows_resume_cites_the_chosen_row() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "dscr")] = page(
        [
            catalog_row("row-obs", "DSCR наблюдённый", concept="dscr.observed"),
            catalog_row("row-lim", "DSCR лимит", concept="dscr.limit", sheet="Limits"),
        ]
    )
    parser.pages[("job-1", "dscr наблюдённый")] = page(
        [catalog_row("row-obs", "DSCR наблюдённый", concept="dscr.observed")]
    )
    parser.observations[("job-1", "row-obs")] = [
        observation("row-obs", "2030", "1.25", "C10", label="DSCR наблюдённый")
    ]
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    paused = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(paused)
    assert "999" not in paused.values["user_question"]
    model.push("plan", plan(["DSCR наблюдённый"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-obs", "2030", "1.25", "C10")])
    )
    await graph.ainvoke(
        Command(resume="Наблюдённый, не лимит"),
        run_config("thread-1"),
        durability="sync",
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["row_key"] == "row-obs"
    assert snap.values["citations"][0]["cell"] == "C10"
    assert "Наблюдённый" in _seen(model, "plan")[-1]["user"]


@pytest.mark.asyncio
async def test_why_after_a_finished_answer_does_not_ask_the_row_again() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    ]
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    model.push("plan", plan(["DSCR"], [{"year": "2030"}], "explain", "precedents"))
    model.push("answer", answer("1.25, вход A1", [cite("row-dscr", "2030", "1.25", "C10")]))
    await graph.ainvoke(
        new_turn_input("А почему?", "job-1"), run_config("thread-1"), durability="sync"
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True
    follow = _seen(model, "plan")[-1]["user"]
    assert "А почему?" in follow
    assert "row-dscr" in follow
    assert parser.observation_calls[-1]["precedent_depth"] == 2


@pytest.mark.asyncio
async def test_two_threads_do_not_mix_citations() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    for job, needle, key, value, cell in (
        ("job-a", "ALPHA", "row-a", "1.11", "A1"),
        ("job-b", "BETA", "row-b", "2.22", "B1"),
    ):
        parser.pages[(job, needle.casefold())] = page([catalog_row(key, needle, concept=needle)])
        parser.observations[(job, key)] = [observation(key, "2030", value, cell, label=needle)]
    graph = _graph(parser, ByQuestion())
    await asyncio.gather(
        graph.ainvoke(new_turn_input("ALPHA?", "job-a"), run_config("thread-a"), durability="sync"),
        graph.ainvoke(new_turn_input("BETA?", "job-b"), run_config("thread-b"), durability="sync"),
    )
    left = await graph.aget_state(run_config("thread-a"))
    right = await graph.aget_state(run_config("thread-b"))
    assert left.values["citations"][0]["value"] == "1.11"
    assert right.values["citations"][0]["value"] == "2.22"
    assert left.values["job_id"] == "job-a"
    assert right.values["job_id"] == "job-b"


@pytest.mark.asyncio
async def test_changed_etag_reloads_and_drops_stale_citations() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    parser.book_etag = "etag-2"
    parser.head_etag = "etag-2"
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    await graph.ainvoke(
        new_turn_input("Какой DSCR в 2030?", "job-1"), run_config("thread-1"), durability="sync"
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["book_etag"] == "etag-2"
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "1.25"
    assert snap.values["citations"][0]["cell"] == "C10"


@pytest.mark.asyncio
async def test_report_not_ready_is_a_partial_answer() -> None:
    parser = FakeParser()
    parser.error = ParserError(409, "report_not_ready")
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is False
    assert "собирается" in snap.values["draft"]
    assert not interrupts_of(snap)
    assert parser.observation_calls == []
    assert model.seen == []


@pytest.mark.asyncio
async def test_bad_plan_json_does_not_search() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    model = ScriptedModel()
    model.push("plan", "bad")
    graph = _graph(parser, model)
    await _run(graph, "Какой?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert "план" in snap.values["draft"]
    assert parser.catalog_calls == []
    assert snap.values["satisfactory"] is False


@pytest.mark.asyncio
async def test_truncated_series_asks_for_a_period() -> None:
    parser = FakeParser()
    parser.force_truncated = True
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"]))
    graph = _graph(parser, model)
    await _run(graph, "Покажи весь DSCR")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "48" in snap.values["user_question"]
    assert _seen(model, "answer") == []


@pytest.mark.asyncio
async def test_crashed_run_continues_with_empty_input() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("answer", "boom")
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    graph = _graph(parser, model)
    with pytest.raises(ModelError):
        await _run(graph, "Какой?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.next
    assert not interrupts_of(snap)
    await graph.ainvoke(None, run_config("thread-1"), durability="sync")
    done = await graph.aget_state(run_config("thread-1"))
    assert done.values["question"] == "Какой?"
    assert done.values["satisfactory"] is True


@pytest.mark.asyncio
async def test_fourth_distinct_gap_stops_the_budget() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    for token in ("91", "92", "93", "94"):
        model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
        model.push("answer", answer(f"Это {token}.", [cite("row-dscr", "2030", token, "C10")]))
    graph = _graph(parser, model)
    await _run(graph, "Какой?")
    assert len(_seen(model, "answer")) == 4
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is False


@pytest.mark.asyncio
async def test_dependents_are_a_second_observation_family() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    child = observation("row-child", "2030", "3", "D4", label="child")
    child["concept_id"] = "cov.child"
    child["formula"] = {
        "precedents": [{"row_key": "prec", "concept_id": "cf.cfads", "label": "CFADS"}]
    }
    parser.observations[("job-1", "row-child")] = [child]
    parser.traces[("job-1", "row-dscr")] = {
        "nodes": [{"row_key": "row-dscr"}, {"row_key": "row-child"}]
    }
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "На что влияет DSCR в 2030?")
    assert [call["row_key"] for call in parser.observation_calls] == ["row-dscr", "row-child"]
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    stored = json.dumps(snap.values["observations"], ensure_ascii=False)
    assert "concept_id" not in stored
    assert "cov.child" not in stored
    assert "cf.cfads" not in stored
    child_row = next(
        item for item in snap.values["observations"] if item.get("row_key") == "row-child"
    )
    assert child_row["label"] == "child"
    assert child_row["value"] == "3"


@pytest.mark.asyncio
async def test_full_operating_label_is_one_catalog_query() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "2026"}]}]
    label = "Operating Income or Loss (EBITDA)"
    row = catalog_row("PF|262", label, concept="pnl.ebitda", axis=["forecast"])
    parser.pages[("job-1", label.casefold())] = page([row])
    parser.pages[("job-1", "operating")] = page(
        [catalog_row("other", "Operating expense")], total=28
    )
    parser.observations[("job-1", "PF|262")] = [
        observation("PF|262", "2026", "10", "C1", label=label)
    ]
    model = ScriptedModel()
    model.push("plan", plan(["Operating"], [{"period_key": "2026"}]))
    graph = _graph(parser, model)
    await _run(graph, "Какой Operating Income or Loss (EBITDA) в 2026?")
    assert [call["q"] for call in parser.catalog_calls] == [label]
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["citations"][0]["row_key"] == "PF|262"
    assert snap.values["citations"][0]["value"] == "10"
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_same_concept_longer_label_stays_a_menu() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "2030"}]}]
    parser.pages[("job-1", "ebitda")] = page(
        [
            catalog_row("short", "EBITDA", concept="pnl.ebitda"),
            catalog_row("long", "Operating Income or Loss (EBITDA)", concept="pnl.ebitda"),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой EBITDA в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "Operating Income or Loss (EBITDA)" in snap.values["user_question"]
    assert parser.observation_calls == []


@pytest.mark.asyncio
async def test_exact_debt_service_drops_the_longer_cfads_label() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
    parser.pages[("job-1", "debt service")] = page(
        [
            catalog_row(
                "Debt|17",
                "Debt service",
                concept="debt.scheduled_payment",
                axis=["forecast"],
            ),
            catalog_row(
                "CFS|14",
                "Cash Flow Available for Debt Service (CFADS)",
                concept="cf.cfads",
                axis=["forecast"],
            ),
        ]
    )
    parser.observations[("job-1", "Debt|17")] = [
        observation("Debt|17", "Y1", "12", "F17", label="Debt service")
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой Debt service в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert parser.observation_calls[0]["row_key"] == "Debt|17"
    assert [item["row_key"] for item in snap.values["citations"]] == ["Debt|17"]


@pytest.mark.asyncio
async def test_exact_ebitda_keeps_the_acronym_line() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "2030"}]}]
    parser.pages[("job-1", "ebitda")] = page(
        [
            catalog_row("short", "EBITDA", concept="pnl.ebitda"),
            catalog_row("empty", "Operating Income or Loss (EBITDA)", concept=None),
            catalog_row("other", "Reported EBITDA", concept="pnl.other"),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой EBITDA в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert "EBITDA [Model]" in question
    assert "Operating Income or Loss (EBITDA)" in question
    assert "Reported EBITDA" not in question
    assert "pnl.ebitda" not in question
    assert parser.observation_calls == []


@pytest.mark.asyncio
async def test_cfads_during_debt_term_stays_beside_the_exact_label() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "2030"}]}]
    parser.pages[("job-1", "cfads")] = page(
        [
            catalog_row("exact", "CFADS", concept="cf.cfads"),
            catalog_row("during", "CFADS during debt term", concept=None),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой CFADS в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert "CFADS [Model]" in question
    assert "cf.cfads" not in question
    assert "CFADS during debt term" in question
    assert parser.observation_calls == []


@pytest.mark.asyncio
async def test_irr_menu_drops_capex_and_keeps_project_irr() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "2030"}]}]
    parser.pages[("job-1", "irr")] = page(
        [
            catalog_row("capex", "CAPEX", concept="cf.capex"),
            catalog_row("project", "Project IRR", concept=None),
            catalog_row("equity", "Equity IRR", concept="val.irr"),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какие IRR есть в модели?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert "Project IRR" in question
    assert "Equity IRR" in question
    assert "CAPEX" not in question
    assert parser.observation_calls == []
    assert [call["q"] for call in parser.catalog_calls] == ["IRR"]


@pytest.mark.asyncio
async def test_compare_uses_both_keys_from_the_question() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y5"}, {"period_key": "Y10"}]}]
    parser.pages[("job-1", "ebitda")] = page(
        [catalog_row("P&L|13", "EBITDA", concept="pnl.ebitda", axis=["forecast"])]
    )
    parser.observations[("job-1", "P&L|13")] = [
        observation("P&L|13", "Y5", "11", "F5", label="EBITDA"),
        observation("P&L|13", "Y10", "22", "O5", label="EBITDA"),
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Сравни EBITDA в Y5 и в Y10.")
    assert parser.observation_calls[0]["period_ids"] == ["Y5", "Y10"]
    snap = await graph.aget_state(run_config("thread-1"))
    assert {item["period_id"] for item in snap.values["citations"]} == {"Y5", "Y10"}
    assert snap.values["satisfactory"] is True


@pytest.mark.asyncio
async def test_unresolved_phase_does_not_load_the_series() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "P&L!r2", "periods": [{"period_key": "Y1"}, {"period_key": "Y40"}]}]
    parser.pages[("job-1", "ebitda")] = page(
        [catalog_row("P&L|13", "EBITDA", concept="pnl.ebitda", axis=["P&L!r2"])]
    )
    parser.observations[("job-1", "P&L|13")] = [
        observation("P&L|13", "Y40", "562668.75679656409", "AS13", label="EBITDA")
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой EBITDA в первый операционный год?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.observation_calls == []
    assert interrupts_of(snap)
    assert "562668.75679656409" not in snap.values["user_question"]


@pytest.mark.asyncio
async def test_sibling_repayment_start_fetches_the_money_row() -> None:
    bare = [{"period_key": "Y1"}, {"period_key": "Y5"}]
    tba = [
        {
            "period_key": "Y1",
            "phase": "construction",
            "flags": {"repayment start date": False},
        },
        {
            "period_key": "Y5",
            "phase": "operation",
            "phase_year": 1,
            "flags": {"repayment start date": True},
        },
    ]
    parser = FakeParser()
    parser.axes = [{"id": "Debt!r3", "periods": bare}, {"id": "TBA!r2", "periods": tba}]
    parser.pages[("job-1", "debt service")] = page(
        [
            catalog_row(
                "Debt|17",
                "Debt service",
                concept="debt.scheduled_payment",
                axis=["Debt!r3"],
            )
        ]
    )
    parser.observations[("job-1", "Debt|17")] = [
        observation("Debt|17", "Y5", "-17099.33", "E17", label="Debt service")
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой Debt service в год начала погашения?")
    assert parser.observation_calls[0]["row_key"] == "Debt|17"
    assert parser.observation_calls[0]["period_ids"] == ["Y5"]
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["citations"][0]["value"] == "-17099.33"


@pytest.mark.asyncio
async def test_scalar_with_a_missing_year_does_not_cite_the_scalar() -> None:
    parser = FakeParser()
    parser.axes = [
        {"id": "timeline", "periods": [{"period_key": "2030"}]},
        {"id": "point", "periods": []},
    ]
    label = "Average Debt Service Coverage Ratio (DSCR)"
    parser.pages[("job-1", label.casefold())] = page(
        [catalog_row("PF|53", label, concept="val.dscr", axis=["point"])]
    )
    parser.observations[("job-1", "PF|53")] = [
        observation("PF|53", "", "1.8617377551507139", "L53", label=label)
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, f"Какой {label} в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.observation_calls == []
    assert interrupts_of(snap)
    assert "1.8617377551507139" not in snap.values["user_question"]


@pytest.mark.asyncio
async def test_params_value_column_is_published_as_a_scalar() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "point", "periods": []}]
    label = "Average Debt Service Coverage Ratio (DSCR)"
    value = "1.8617377551507139"
    parser.pages[("job-1", label.casefold())] = page(
        [catalog_row("PF|53", label, concept="val.dscr", axis=["point"])]
    )
    parser.observations[("job-1", "PF|53")] = [
        observation("PF|53", "value", value, "L53", label=label)
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, f"Какой {label}?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["draft"] == f"{label}: {value}"
    assert "в value" not in snap.values["draft"]
    assert snap.values["citations"][0]["period_id"] == ""
    assert snap.values["citations"][0]["value"] == value
    assert snap.values["satisfactory"] is True


@pytest.mark.asyncio
async def test_cache_string_is_copied_with_the_scale_word() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y40"}]}]
    value = "562668.75679656409"
    parser.pages[("job-1", "ebitda")] = page(
        [catalog_row("P&L|13", "EBITDA", concept="pnl.ebitda", axis=["forecast"])]
    )
    parser.observations[("job-1", "P&L|13")] = [
        observation(
            "P&L|13",
            "Y40",
            value,
            "AS13",
            scale_factor=1000,
            scale="k",
            label="EBITDA",
        )
    ]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой EBITDA в Y40?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert value in snap.values["draft"]
    assert "тыс." in snap.values["draft"]
    assert snap.values["citations"][0]["value"] == value
    assert snap.values["satisfactory"] is True


@pytest.mark.asyncio
async def test_scale_gap_on_a_model_answer_still_publishes() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "2030", "12.5", "C10", scale_factor=1000, scale="k", label="DSCR")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("answer", answer("DSCR в 2030: 12.5.", [cite("row-dscr", "2030", "12.5", "C10")]))
    graph = _graph(parser, model)
    await _run(graph, "Какой?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "12.5"
    assert snap.values["gaps"] == []


@pytest.mark.asyncio
async def test_longer_ebitda_stays_in_the_menu() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "2030"}]}]
    parser.pages[("job-1", "ebitda")] = page(
        [
            catalog_row("short", "EBITDA", concept="pnl.ebitda", sheet="PF Model"),
            catalog_row(
                "long",
                "Operating Income or Loss (EBITDA)",
                concept=None,
                sheet="PF Model",
            ),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой EBITDA в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert "Operating Income or Loss (EBITDA)" in question
    assert "EBITDA" in question
    assert "PF Model" in question
    assert "pnl.ebitda" not in question
    assert parser.observation_calls == []


@pytest.mark.asyncio
async def test_path_only_irr_row_stays_out_of_the_menu() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "irr")] = page(
        [
            catalog_row("project", "Project IRR", concept=None, label_path=[]),
            catalog_row("equity", "Equity IRR", concept=None, label_path=["Equity"]),
            catalog_row("capex", "CAPEX", concept="cf.capex", label_path=["Project IRR"]),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какие IRR есть в модели?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert "Project IRR" in question
    assert "Equity IRR" in question
    assert "Equity" in question
    assert "CAPEX" not in question
    assert not snap.values.get("citations")


@pytest.mark.asyncio
async def test_same_label_uses_the_section_heading() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "project irr")] = page(
        [
            catalog_row(
                "heading",
                "Project IRR",
                sheet="Ratios",
                label_path=["Returns"],
                kind="abstract",
            ),
            catalog_row(
                "value",
                "Project IRR",
                sheet="Ratios",
                label_path=[],
                kind="fact",
            ),
            catalog_row(
                "other",
                "Project IRR",
                sheet="Ratios",
                label_path=[],
                kind="abstract",
            ),
        ]
    )
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "Какой Project IRR?")
    question = (await graph.aget_state(run_config("thread-1"))).values["user_question"]
    assert "Returns" in question
    assert "значение" in question
    assert "заголовок" in question


@pytest.mark.asyncio
async def test_three_labels_cite_three_rows_without_the_model() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    rows = (
        ("CFADS", "cfs", "10"),
        ("Total debt service", "debt", "20"),
        ("DSCR", "dscr", "30"),
    )
    for needle, key, value in rows:
        parser.pages[("job-1", needle.casefold())] = page(
            [catalog_row(key, needle, axis=["forecast"])]
        )
        parser.observations[("job-1", key)] = [observation(key, "2030", value, "C1", label=needle)]
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Покажи CFADS, Total debt service и DSCR в 2030")
    snap = await graph.aget_state(run_config("thread-1"))
    assert {call["q"] for call in parser.catalog_calls} == {
        "CFADS",
        "Total debt service",
        "DSCR",
    }
    assert {item["row_key"] for item in snap.values["citations"]} == {"cfs", "debt", "dscr"}
    assert {item["period_id"] for item in snap.values["citations"]} == {"2030"}
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_two_forks_pause_without_numbers() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "cfads")] = page(
        [
            catalog_row("a", "CFADS"),
            catalog_row("b", "CFADS during debt term"),
        ]
    )
    parser.pages[("job-1", "ebitda")] = page(
        [
            catalog_row("c", "EBITDA"),
            catalog_row("d", "Operating Income or Loss (EBITDA)", concept=None),
        ]
    )
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Сравни CFADS и EBITDA в 2030")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    question = snap.values["user_question"]
    assert "CFADS" in question
    assert "EBITDA" in question
    assert parser.observation_calls == []
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_five_needles_asks_to_narrow() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    graph = _graph(parser, ScriptedModel())
    await _run(graph, "CFADS, EBITDA, DSCR, IRR, Debt")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert snap.values["user_question"] == "Назовите не больше четырёх."
    assert parser.catalog_calls == []


@pytest.mark.asyncio
async def test_cover_question_pauses_without_catalog() -> None:
    parser = FakeParser()
    parser.summary = {"sheets": ["P&L", "CFS", "Debt"], "marker": "passport"}
    parser.axes = year_axes()
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Какая общая картина?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert parser.catalog_calls == []
    assert "P&L" in question
    assert "CFS" in question
    assert "подписей" in question
    assert "CFADS" not in question
    assert "DSCR" not in question
    assert "IRR" not in question
    assert not any(char.isdigit() for char in question)
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_cover_phrase_with_two_labels_is_a_compose() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "point", "periods": []}]
    left = "Average Debt Service Coverage Ratio (DSCR)"
    right = "Minimum Debt Service Coverage Ratio (DSCR)"
    parser.pages[("job-1", left.casefold())] = page([catalog_row("avg", left, axis=["point"])])
    parser.pages[("job-1", right.casefold())] = page([catalog_row("min", right, axis=["point"])])
    parser.observations[("job-1", "avg")] = [observation("avg", "", "1.8", "L53", label=left)]
    parser.observations[("job-1", "min")] = [observation("min", "", "1.4", "L54", label=right)]
    graph = _graph(parser, ScriptedModel())
    await _run(graph, f"Ключевые показатели: {left} и {right}")
    snap = await graph.aget_state(run_config("thread-1"))
    assert {item["row_key"] for item in snap.values["citations"]} == {"avg", "min"}
    assert all(call["q"] != "ключевые показатели" for call in parser.catalog_calls)
    assert {call["q"] for call in parser.catalog_calls} == {left, right}


@pytest.mark.asyncio
async def test_cover_reply_is_planned_in_code() -> None:
    parser = FakeParser()
    parser.summary = {"sheets": ["P&L", "CFS"], "marker": "passport"}
    parser.axes = year_axes()
    parser.pages[("job-1", "cfads")] = page([catalog_row("cfs", "CFADS", axis=["forecast"])])
    parser.pages[("job-1", "ebitda")] = page([catalog_row("pnl", "EBITDA", axis=["forecast"])])
    parser.observations[("job-1", "cfs")] = [observation("cfs", "2030", "10", "A1", label="CFADS")]
    parser.observations[("job-1", "pnl")] = [observation("pnl", "2030", "20", "A2", label="EBITDA")]
    model = ScriptedModel()
    graph = _graph(parser, model)
    await _run(graph, "Какая общая картина?")
    await graph.ainvoke(
        Command(resume="CFADS и EBITDA в 2030"),
        run_config("thread-1"),
        durability="sync",
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert {item["row_key"] for item in snap.values["citations"]} == {"cfs", "pnl"}
    assert {item["period_id"] for item in snap.values["citations"]} == {"2030"}
    assert _seen(model, "plan") == []
