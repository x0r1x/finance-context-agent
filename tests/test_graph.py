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
    await _run(graph, "Какой долг?")
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
    model.push("critic", {"gaps": []})
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
    model = ScriptedModel()
    model.push("plan", plan(["EBITDA", "DSCR"], [], "compose", "none"))
    graph = _graph(parser, model)
    await _run(graph, "Сравни EBITDA и DSCR")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "Какую строку" in snap.values["user_question"]
    assert parser.observation_calls == []


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
    model.push("critic", {"gaps": ["указание валюты"]})
    graph = _graph(parser, model)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["gaps"] == []
    assert snap.values["citations"][0]["row_key"] == "P&L|13|P&L!r2"
    assert snap.values["citations"][0]["period_id"] == "Y1"
    assert snap.values["citations"][0]["value"] == "0"
    assert snap.values["citations"][0]["value_status"] == "zero_explicit"
    assert len(_seen(model, "plan")) == 1


@pytest.mark.asyncio
async def test_question_prefix_still_finds_the_named_period() -> None:
    parser = FakeParser()
    parser.axes = [{"id": "forecast", "periods": [{"period_key": "Y1"}]}]
    row = catalog_row("P&L|13|P&L!r2", "EBITDA", concept="pnl.ebitda")
    parser.pages[("job-1", "ebitda")] = page([row])
    parser.observations[("job-1", "P&L|13|P&L!r2")] = [
        observation(
            "P&L|13|P&L!r2", "Y1", "0", "C10", status="zero_explicit", label="EBITDA"
        )
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
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    queries = [call["q"] for call in parser.catalog_calls]
    assert queries[0] == "Какой EBITDA"
    assert "EBITDA" in queries
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
    assert "Debt" not in queries
    assert "Какой долг" in queries
    assert "долг" in queries
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
    model.push("critic", {"gaps": []})
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
    model.push("critic", {"gaps": []})
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
    model.push("critic", {"gaps": []})
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
    assert "context.json" not in _seen(model, "answer")[0]["user"]
    assert "CATALOG_PAGE" not in _seen(model, "plan")[0]["user"]


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
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "DSCR в 2030 и в первом году")
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
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    assert len(_seen(model, "answer")) == 2
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "1.25"


@pytest.mark.asyncio
async def test_why_at_depth_zero_retries_at_depth_two_without_asking() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push("plan", plan(["DSCR"], [{"year": "2030"}], "explain", "precedents"))
    model.push("answer", answer("1.25", [cite("row-dscr", "2030", "1.25", "C10")]))
    model.push("answer", answer("1.25", [cite("row-dscr", "2030", "1.25", "C10")]))
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Почему DSCR в 2030 равен 1.25?")
    assert [call["precedent_depth"] for call in parser.observation_calls] == [0, 2]
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values.get("awaiting") in ("", None)
    assert snap.values["satisfactory"] is True


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
    await _run(graph, "Какой DSCR в 2030?")
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
        observation("row-dscr", "2030", "", "C10", status="empty")
    ]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("Ячейка empty.", [cite("row-dscr", "2030", None, "C10", status="empty")])
    )
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
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
    await _run(graph, "Где ZZZ?")
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
    model = ScriptedModel()
    model.push("plan", plan(["Выручка", "EBITDA"], [{"year": "2030"}], "compose"))
    model.push(
        "answer",
        answer(
            "Выручка 10 и EBITDA 4.",
            [cite("row-rev", "2030", "10", "B2"), cite("row-ebitda", "2030", "4", "B3")],
        ),
    )
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Выручка и EBITDA в 2030")
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
    model.push("critic", {"gaps": []})
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
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    paused = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(paused)
    assert "999" not in paused.values["user_question"]
    model.push("plan", plan(["DSCR наблюдённый"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-obs", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
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
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    model.push("plan", plan(["DSCR"], [{"year": "2030"}], "explain", "precedents"))
    model.push("answer", answer("1.25", [cite("row-dscr", "2030", "1.25", "C10")]))
    model.push("critic", {"gaps": []})
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
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "Какой DSCR в 2030?")
    parser.book_etag = "etag-2"
    parser.head_etag = "etag-2"
    model.push("plan", plan(["DSCR"], [{"year": "2030"}]))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    await graph.ainvoke(
        new_turn_input("Какой DSCR в 2030?", "job-1"), run_config("thread-1"), durability="sync"
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["book_etag"] == "etag-2"
    assert snap.values["satisfactory"] is True
    follow = _seen(model, "plan")[-1]["user"]
    assert '"prior_citations": []' in follow


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
    await _run(graph, "Какой DSCR?")
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
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    with pytest.raises(ModelError):
        await _run(graph, "Какой DSCR в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.next
    assert not interrupts_of(snap)
    await graph.ainvoke(None, run_config("thread-1"), durability="sync")
    done = await graph.aget_state(run_config("thread-1"))
    assert done.values["question"] == "Какой DSCR в 2030?"
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
    await _run(graph, "Какой DSCR в 2030?")
    assert len(_seen(model, "answer")) == 4
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is False


@pytest.mark.asyncio
async def test_dependents_are_a_second_observation_family() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    parser.observations[("job-1", "row-child")] = [
        observation("row-child", "2030", "3", "D4", label="child")
    ]
    parser.traces[("job-1", "row-dscr")] = {
        "nodes": [{"row_key": "row-dscr"}, {"row_key": "row-child"}]
    }
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"], [{"year": "2030"}], "lookup", "dependents"))
    model.push(
        "answer", answer("DSCR в 2030 равен 1.25.", [cite("row-dscr", "2030", "1.25", "C10")])
    )
    model.push("critic", {"gaps": []})
    graph = _graph(parser, model)
    await _run(graph, "На что влияет DSCR в 2030?")
    assert [call["row_key"] for call in parser.observation_calls] == ["row-dscr", "row-child"]
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
