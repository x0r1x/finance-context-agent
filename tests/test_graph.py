import asyncio
import json
import logging

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from finance_context_agent.catalog import choice_question, narrow_page
from finance_context_agent.clients.llm import ModelError
from finance_context_agent.clients.parser import ParserError
from finance_context_agent.graph import build_graph
from finance_context_agent.questions import (
    ACT_FLOOR,
    ACT_GAP,
    ACTS,
    book_overview,
    books_reply,
    chitchat_reply,
    goal_question,
    question_needles,
)
from finance_context_agent.session import interrupts_of
from finance_context_agent.settings import Settings
from finance_context_agent.turn import new_turn_input, run_config
from tests.fakes import (
    ByQuestion,
    FakeEmbed,
    FakeParser,
    FakeRanker,
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


def _graph(
    parser: FakeParser,
    model,
    settings: Settings | None = None,
    ranker: FakeRanker | None = None,
    embedder: FakeEmbed | None = None,
) -> object:
    return build_graph(
        parser, model, InMemorySaver(), settings, ranker or FakeRanker(), embedder
    )


async def _run(graph, question: str, thread: str = "thread-1", job: str = "job-1"):
    await graph.ainvoke(new_turn_input(question, job), run_config(thread), durability="sync")


def _bind(parser: FakeParser, needle: str, rows: list[dict], job: str = "job-1") -> None:
    parser.axes = year_axes()
    parser.pages[(job, needle.casefold())] = page(rows)


def _seen(model: ScriptedModel, role: str) -> list[dict[str, str]]:
    return [item for item in model.seen if item["role"] == role]


def _act(embed: FakeEmbed, act: str, top: float = 0.90, second: float = 0.20) -> None:
    """One act above both floors. The other six share the lower cosine."""
    scores = {name: second for name in ACTS}
    scores[act] = top
    embed.push(scores)


def _score(ranker: FakeRanker, key: str, score: float = 0.9) -> None:
    weights = {
        "books": 0.02,
        "book": 0.02,
        "intro": 0.02,
        "r0": 0.01,
        "r1": 0.01,
        "1": 0.01,
        "2": 0.01,
    }
    weights[key] = score
    ranker.push(key, weights)


def _offer_both(ranker: FakeRanker) -> None:
    ranker.push(
        "r0",
        {"r0": 0.4, "r1": 0.35, "books": 0.05, "book": 0.04, "intro": 0.03},
    )


def _slim(values: dict) -> None:
    blob = json.dumps(values, ensure_ascii=False)
    assert "CATALOG_PAGE" not in blob
    assert "DISCARD_ROW" not in blob
    assert "page_marker" not in blob
    assert values["summary"]["marker"] == "passport"
    assert values["axes"][0]["periods"]


def _menu_text(needle, rows, total, *, narrow: bool):
    catalog = page(rows, total)
    if narrow:
        mention, shown = narrow_page(needle, catalog)
    else:
        mention, shown = needle, catalog
    keys = [row["row_key"] for row in shown["rows"]]
    return keys, choice_question([(mention, shown)])


@pytest.mark.parametrize(
    ("needle", "rows", "total", "narrow", "kept", "present", "absent"),
    [
        pytest.param(
            "Debt service",
            [
                catalog_row("Debt|17", "Debt service", concept="debt.scheduled_payment"),
                catalog_row(
                    "CFS|14",
                    "Cash Flow Available for Debt Service (CFADS)",
                    concept="cf.cfads",
                ),
            ],
            None,
            True,
            ["Debt|17"],
            ["Debt service"],
            ["CFADS", "Cash Flow Available"],
            id="exact-debt-service-drops-cfads",
        ),
        pytest.param(
            "EBITDA",
            [
                catalog_row("short", "EBITDA", concept="pnl.ebitda"),
                catalog_row("empty", "Operating Income or Loss (EBITDA)", concept=None),
                catalog_row("other", "Reported EBITDA", concept="pnl.other"),
            ],
            None,
            True,
            ["short", "empty"],
            ["№1 EBITDA", "Operating Income or Loss (EBITDA)", "Лист Model"],
            ["Reported EBITDA", "["],
            id="exact-ebitda-keeps-acronym",
        ),
        pytest.param(
            "CFADS",
            [
                catalog_row("exact", "CFADS", concept="cf.cfads"),
                catalog_row("during", "CFADS during debt term", concept=None),
            ],
            None,
            True,
            ["exact", "during"],
            ["№1 CFADS", "CFADS during debt term", "Лист Model"],
            [],
            id="cfads-prefix-keeps-debt-term",
        ),
        pytest.param(
            "IRR",
            [
                catalog_row("capex", "CAPEX", concept="cf.capex"),
                catalog_row("project", "Project IRR", concept=None),
                catalog_row("equity", "Equity IRR", concept="val.irr"),
            ],
            None,
            True,
            ["project", "equity"],
            ["Project IRR", "Equity IRR"],
            ["CAPEX"],
            id="irr-keeps-project-and-equity",
        ),
        pytest.param(
            "EBITDA",
            [
                catalog_row("short", "EBITDA", concept="pnl.ebitda"),
                catalog_row("long", "Operating Income or Loss (EBITDA)", concept="pnl.ebitda"),
            ],
            None,
            True,
            ["short", "long"],
            ["Operating Income or Loss (EBITDA)"],
            [],
            id="same-concept-longer-stays",
        ),
        pytest.param(
            "EBITDA",
            [
                catalog_row("short", "EBITDA", concept="pnl.ebitda", sheet="PF Model"),
                catalog_row(
                    "long",
                    "Operating Income or Loss (EBITDA)",
                    concept=None,
                    sheet="PF Model",
                ),
            ],
            None,
            True,
            ["short", "long"],
            ["Operating Income or Loss (EBITDA)", "EBITDA", "PF Model"],
            [],
            id="longer-ebitda-stays-with-sheet",
        ),
        pytest.param(
            "IRR",
            [
                catalog_row("project", "Project IRR", concept=None, label_path=[]),
                catalog_row("equity", "Equity IRR", concept=None, label_path=["Equity"]),
                catalog_row("capex", "CAPEX", concept="cf.capex", label_path=["Project IRR"]),
                catalog_row("hidden", "WACC", concept="returns.irr"),
            ],
            None,
            True,
            ["project", "equity"],
            ["Project IRR", "Equity IRR", "Раздел Equity"],
            ["CAPEX", "WACC", "["],
            id="path-or-concept-only-stays-out",
        ),
        pytest.param(
            "Project IRR",
            [
                catalog_row(
                    "heading",
                    "Project IRR",
                    sheet="Ratios",
                    label_path=["Returns"],
                    kind="abstract",
                ),
                catalog_row("value", "Project IRR", sheet="Ratios", label_path=[], kind="fact"),
                catalog_row("other", "Project IRR", sheet="Ratios", label_path=[], kind="abstract"),
            ],
            None,
            True,
            ["heading", "value", "other"],
            ["Returns", "значение", "заголовок", "Лист Ratios"],
            ["["],
            id="same-label-uses-section-heading",
        ),
        pytest.param(
            "DSCR",
            [catalog_row("a", "Observed"), catalog_row("b", "Limit")],
            None,
            False,
            ["a", "b"],
            ["Какую строку"],
            ["из "],
            id="full-page-does-not-say-iz",
        ),
        pytest.param(
            "Debt",
            [catalog_row(f"row-{index}", f"Debt {index}") for index in range(8)],
            26,
            True,
            [f"row-{index}" for index in range(8)],
            ["8 из 26", "Какую строку"],
            [],
            id="short-page-says-how-many",
        ),
    ],
)
def test_own_label_menu(needle, rows, total, narrow, kept, present, absent) -> None:
    keys, text = _menu_text(needle, rows, total, narrow=narrow)
    assert keys == kept
    for piece in present:
        assert piece in text
    for piece in absent:
        assert piece not in text
    for row in rows:
        concept = row.get("concept_id")
        if concept:
            assert concept not in text


_Y5_Y10 = [{"id": "forecast", "periods": [{"period_key": "Y5"}, {"period_key": "Y10"}]}]
_YEAR_2026 = [{"id": "forecast", "periods": [{"period_key": "2026"}]}]


@pytest.mark.parametrize(
    ("question", "axes", "expected", "forbidden"),
    [
        pytest.param("Какой долг?", year_axes(), ["долг"], "Debt", id="debt-is-not-rewritten"),
        pytest.param(
            "Сравни EBITDA в Y5 и в Y10",
            _Y5_Y10,
            ["EBITDA"],
            "Y5",
            id="period-key-is-not-a-needle",
        ),
        pytest.param(
            "Cash in Bank",
            year_axes(),
            ["Cash in Bank"],
            "в",
            id="in-stays-inside-the-label",
        ),
        pytest.param(
            "Какие IRR есть в модели?",
            year_axes(),
            ["IRR"],
            "модели",
            id="irr-question-drops-the-function-words",
        ),
        pytest.param(
            "Какой Operating Income or Loss (EBITDA) в 2026?",
            _YEAR_2026,
            ["Operating Income or Loss (EBITDA)"],
            "2026",
            id="long-label-is-one-needle",
        ),
        pytest.param(
            "CFADS, EBITDA, DSCR, IRR, Debt",
            year_axes(),
            ["CFADS", "EBITDA", "DSCR", "IRR", "Debt"],
            "четырёх",
            id="five-needles-stay-five",
        ),
        pytest.param(
            "Какая общая картина?",
            year_axes(),
            [],
            "картина",
            id="cover-phrase-yields-no-metric-needles",
        ),
    ],
)
def test_needles_from_the_question(question, axes, expected, forbidden) -> None:
    found = question_needles(question, axes)
    assert found == expected
    assert forbidden not in found
    assert forbidden not in " ".join(found)


@pytest.mark.asyncio
async def test_short_catalog_page_says_how_many_labels_are_shown() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    rows = [catalog_row(f"row-{index}", f"Debt {index}") for index in range(8)]
    parser.pages[("job-1", "debt")] = page(rows, total=26)
    embedder = FakeEmbed()
    _act(embedder, "figure")
    graph = _graph(parser, ScriptedModel(), embedder=embedder)
    await _run(graph, "Какой?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert parser.catalog_calls == []
    assert "Такой строки нет" in snap.values["user_question"]
    assert "8 из 26" not in snap.values["user_question"]


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert parser.catalog_calls == []
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "0"
    assert snap.values["citations"][0]["value_status"] == "zero_explicit"
    assert "999" not in snap.values["draft"]


@pytest.mark.asyncio
async def test_compose_keeps_a_missing_metric() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "ebitda")] = page([catalog_row("row-ebitda", "EBITDA")])
    parser.observations[("job-1", "row-ebitda")] = [
        observation("row-ebitda", "2030", "10", "B1", label="EBITDA")
    ]
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Сравни EBITDA и DSCR")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert parser.catalog_calls == []
    assert [call["row_key"] for call in parser.observation_calls] == ["row-ebitda"]
    assert {item["row_key"] for item in snap.values["citations"]} == {"row-ebitda"}
    assert "DSCR" not in (snap.values.get("user_question") or "")
    assert "Какую строку" not in (snap.values.get("draft") or "")


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.catalog_calls == []
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
    embedder = FakeEmbed()
    _act(embedder, "figure")
    graph = _graph(parser, model, embedder=embedder)
    await _run(graph, "Какой долг?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.catalog_calls == []
    assert "Debt" not in snap.values["user_question"]
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert parser.observation_calls == []
    assert parser.context_calls == []
    assert _seen(model, "about") == []
    assert _seen(model, "plan") == []


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "EBITDA Y1")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.catalog_calls == []
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Cash in Bank")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.catalog_calls == []
    assert [row["row_key"] for row in snap.values["selected"]] == ["row-cash"]
    assert snap.values["citations"][0]["value"] == "5"


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой DSCR в 2030?")
    assert _seen(model, "answer") == []
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "1.25"
    assert "99" not in snap.values["draft"]


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Как считается DSCR в 2030?")
    assert parser.catalog_calls == []
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
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "Y1", "1.25", "C10")]
    model = ScriptedModel()
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой DSCR в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is False
    assert snap.values["draft"] == "Подтверждённого числа в срезе нет."
    assert "Не хватает" not in snap.values["draft"]
    assert snap.values["gaps"] == []
    assert _seen(model, "answer") == []
    assert _seen(model, "view") == []


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    parser.observations[("job-1", "row-lim")] = [
        observation("row-lim", "2029", "1.10", "C9", label="DSCR лимит"),
        observation("row-lim", "2030", "1.25", "C10", label="DSCR лимит"),
    ]
    model = ScriptedModel()
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _offer_both(ranker)
    graph = _graph(parser, model, ranker=ranker, embedder=embedder)
    await _run(graph, "Какой DSCR в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert parser.observation_calls == []
    question = snap.values["user_question"]
    assert "Какую строку" in question
    assert "999" not in question
    assert "1.25" not in question
    _slim(snap.values)
    assert {row["row_key"] for row in snap.values["offers"]} == {"row-obs", "row-lim"}
    assert "concept_id" not in snap.values["offers"][0]
    searched = len(parser.catalog_calls)
    menu = snap.values["user_question"]
    assert "Напишите номер" in menu
    assert "№1" in menu
    assert "№2" in menu
    assert "Лист" in menu
    assert "[" not in menu
    heard = len(ranker.calls)
    for _ in range(3):
        await graph.ainvoke(
            Command(resume="Какой DSCR в 2030?"),
            run_config("thread-1"),
            durability="sync",
        )
        again = await graph.aget_state(run_config("thread-1"))
        assert interrupts_of(again)
        assert again.values["user_question"] == menu
        assert "DSCR лимит" in again.values["user_question"]
        assert "DSCR наблюдённый" in again.values["user_question"]
        assert "Не нашёл такую строку" not in again.values["user_question"]
        assert "Не смог выбрать строку" not in again.values["user_question"]
        assert again.values["clarify_rounds"] == 0
        assert parser.observation_calls == []
        assert len(parser.catalog_calls) == searched
        assert _seen(model, "plan") == []
        assert len(ranker.calls) == heard
    before_number = len(ranker.calls)
    await graph.ainvoke(
        Command(resume="2"),
        run_config("thread-1"),
        durability="sync",
    )
    chosen = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(chosen)
    assert chosen.values["satisfactory"] is True
    assert "clarify" not in chosen.values["gaps"]
    assert "Не смог выбрать строку" not in chosen.values["draft"]
    assert chosen.values["citations"][0]["row_key"] == "row-lim"
    assert parser.observation_calls[0]["row_key"] == "row-lim"
    assert parser.observation_calls[0]["period_ids"] == ["2030"]
    assert len(parser.catalog_calls) == searched
    assert _seen(model, "plan") == []
    assert len(ranker.calls) == before_number
    _offer_both(ranker)
    await _run(graph, "Какой DSCR в 2030?", thread="thread-2")
    _act(embedder, "figure")
    await graph.ainvoke(
        Command(resume="нет такой подписи"),
        run_config("thread-2"),
        durability="sync",
    )
    first_miss = await graph.aget_state(run_config("thread-2"))
    assert interrupts_of(first_miss)
    assert "Такой строки нет" in first_miss.values["user_question"]
    _act(embedder, "figure")
    await graph.ainvoke(
        Command(resume="нет такой подписи"),
        run_config("thread-2"),
        durability="sync",
    )
    closed = await graph.aget_state(run_config("thread-2"))
    assert not interrupts_of(closed)
    assert "Не смог выбрать строку" in closed.values["draft"]
    assert "clarify" in closed.values["gaps"]
    _offer_both(ranker)
    await _run(graph, "Какой DSCR в 2030?", thread="thread-phrase")
    _score(ranker, "r0")
    await graph.ainvoke(
        Command(resume="DSCR лимит на листе Limits"),
        run_config("thread-phrase"),
        durability="sync",
    )
    phrased = await graph.aget_state(run_config("thread-phrase"))
    assert not interrupts_of(phrased)
    assert phrased.values["citations"][0]["row_key"] == "row-lim"
    assert "clarify" not in phrased.values["gaps"]
    assert _seen(model, "plan") == []
    last = ranker.calls[-1]
    assert list(last["criteria"].values()) == ["DSCR лимит"]
    blob = json.dumps(last, ensure_ascii=False)
    assert "concept_id" not in blob
    assert "row_key" not in blob
    assert "999" not in blob
    _offer_both(ranker)
    await _run(graph, "Какой DSCR в 2030?", thread="thread-label")
    labeled = len(ranker.calls)
    await graph.ainvoke(
        Command(resume="DSCR лимит"),
        run_config("thread-label"),
        durability="sync",
    )
    exact = await graph.aget_state(run_config("thread-label"))
    assert not interrupts_of(exact)
    assert exact.values["citations"][0]["row_key"] == "row-lim"
    assert len(ranker.calls) == labeled
    _offer_both(ranker)
    await _run(graph, "Какой DSCR в 2030?", thread="thread-mark")
    marked = len(ranker.calls)
    await graph.ainvoke(
        Command(resume="№2"),
        run_config("thread-mark"),
        durability="sync",
    )
    mark = await graph.aget_state(run_config("thread-mark"))
    assert not interrupts_of(mark)
    assert mark.values["citations"][0]["row_key"] == "row-lim"
    assert len(ranker.calls) == marked
    _offer_both(ranker)
    await _run(graph, "Какой DSCR?", thread="thread-period")
    await graph.ainvoke(
        Command(resume="2"),
        run_config("thread-period"),
        durability="sync",
    )
    period = await graph.aget_state(run_config("thread-period"))
    assert not interrupts_of(period)
    assert period.values["satisfactory"] is True
    assert "Назовите период." not in (period.values.get("draft") or "")
    assert "Не смог выбрать строку" not in (period.values.get("draft") or "")
    assert "clarify" not in period.values.get("gaps", [])
    cited = [
        (item["period_id"], item["value"], item["cell"]) for item in period.values["citations"]
    ]
    assert cited == [("2029", "1.10", "C9"), ("2030", "1.25", "C10")]
    assert "DSCR лимит в 2029: 1.10" in period.values["draft"]
    assert "DSCR лимит в 2030: 1.25" in period.values["draft"]
    catalog_before = len(parser.catalog_calls)
    plans_before = len(_seen(model, "plan"))
    heard = len(ranker.calls)
    await graph.ainvoke(
        new_turn_input("в 2030", "job-1"),
        run_config("thread-period"),
        durability="sync",
    )
    narrowed = await graph.aget_state(run_config("thread-period"))
    assert not interrupts_of(narrowed)
    assert "Какую строку" not in (narrowed.values.get("draft") or "")
    assert narrowed.values["satisfactory"] is True
    assert [(item["period_id"], item["value"]) for item in narrowed.values["citations"]] == [
        ("2030", "1.25"),
    ]
    assert "2029" not in narrowed.values["draft"]
    assert len(parser.catalog_calls) == catalog_before
    assert len(_seen(model, "plan")) == plans_before
    assert len(ranker.calls) == heard


@pytest.mark.asyncio
async def test_two_misses_ask_that_the_row_is_missing() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.pages[("job-1", "dscr")] = page([catalog_row("row-dscr", "DSCR")])
    parser.pages[("job-1", "capex")] = page([catalog_row("row-capex", "CAPEX")])
    parser.observations[("job-1", "row-dscr")] = [
        observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    ]
    parser.observations[("job-1", "row-capex")] = [
        observation("row-capex", "2030", "5", "C2", label="CAPEX")
    ]
    model = ScriptedModel()
    ranker = FakeRanker()
    embedder = FakeEmbed()
    graph = _graph(parser, model, ranker=ranker, embedder=embedder)
    _act(embedder, "unclear")
    await _run(graph, "ZZZ?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "Уточните:" in snap.values["user_question"]
    assert parser.catalog_calls == []
    assert parser.observation_calls == []
    assert _seen(model, "plan") == []
    assert _seen(model, "about") == []
    assert parser.context_calls == []
    parser.context = _book_document()
    _act(embedder, "overview")
    await graph.ainvoke(
        Command(resume="Расскажи по модель ! дай саммари по ней"),
        run_config("thread-1"),
        durability="sync",
    )
    resumed = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(resumed)
    assert resumed.values["satisfactory"] is True
    assert resumed.values["gaps"] == []
    assert "CPI" in resumed.values["draft"]
    assert "Не смог выбрать строку" not in resumed.values["draft"]
    heard = len(parser.context_calls)
    parser.summary = {
        "source_filename": "packt-project-finance.xlsx",
        "formula_count": 99,
        "sheets": ["A", "B"],
        "marker": "passport",
    }
    parser.jobs = [{"job_id": "job-1", "source_filename": "packt-project-finance.xlsx"}]
    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("Привет", "job-1"),
        run_config("thread-chat"),
        durability="sync",
    )
    greeting = await graph.aget_state(run_config("thread-chat"))
    introduction = chitchat_reply()
    asked = goal_question("packt-project-finance.xlsx")
    assert "finance-context-agent" in introduction
    assert "формулу из кэша" in introduction
    assert interrupts_of(greeting)
    assert greeting.values["pending"] == "goal"
    assert greeting.values["job_id"] == "job-1"
    assert greeting.values["user_question"] == asked
    assert "99" not in greeting.values["user_question"]
    assert "Лист" not in greeting.values["user_question"]
    assert "Готовые книги:" not in greeting.values["user_question"]
    assert "Готовые книги:" not in (greeting.values.get("draft") or "")
    assert parser.observation_calls == []
    assert len(parser.context_calls) == heard
    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("что ты умеешь?", "job-1"),
        run_config("thread-able"),
        durability="sync",
    )
    able = await graph.aget_state(run_config("thread-able"))
    assert interrupts_of(able)
    assert able.values["pending"] == "goal"
    assert able.values["user_question"] == asked
    assert len(parser.context_calls) == heard
    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("Спасибо", "job-1"),
        run_config("thread-thanks"),
        durability="sync",
    )
    thanks = await graph.aget_state(run_config("thread-thanks"))
    assert interrupts_of(thanks)
    assert thanks.values["pending"] == "goal"
    assert thanks.values["user_question"] == asked
    assert "CPI" not in thanks.values["user_question"]
    assert len(parser.context_calls) == heard
    assert parser.observation_calls == []
    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("Привет"),
        run_config("thread-bare"),
        durability="sync",
    )
    bare = await graph.aget_state(run_config("thread-bare"))
    assert interrupts_of(bare)
    assert "Какую книгу открыть?" in bare.values["user_question"]
    assert "packt-project-finance.xlsx" in bare.values["user_question"]
    assert bare.values.get("job_id") in (None, "")
    assert bare.values.get("draft") != introduction
    assert len(parser.context_calls) == heard
    parser.jobs = [
        {"job_id": "job-1", "source_filename": "packt-project-finance.xlsx"},
        {"job_id": "job-2", "source_filename": "rvi-project-finance.xlsx"},
    ]
    _act(embedder, "files")
    listed = len(parser.list_calls)
    assert ranker.calls == []
    await graph.ainvoke(
        new_turn_input("какие данные у тебя есть?", "job-1"),
        run_config("thread-books"),
        durability="sync",
    )
    inventory = await graph.aget_state(run_config("thread-books"))
    assert not interrupts_of(inventory)
    assert inventory.values["draft"] == books_reply(parser.jobs)
    assert "packt-project-finance.xlsx" in inventory.values["draft"]
    assert "rvi-project-finance.xlsx" in inventory.values["draft"]
    assert "Такой строки нет" not in inventory.values["draft"]
    assert not any(char.isdigit() for char in inventory.values["draft"])
    assert len(parser.context_calls) == heard
    assert len(parser.list_calls) == listed + 1
    assert ranker.calls == []
    _act(embedder, "unclear")
    await _run(graph, "QQQ?", thread="thread-leave")
    searched = len(parser.catalog_calls)
    _act(embedder, "greet")
    await graph.ainvoke(
        Command(resume="привет"),
        run_config("thread-leave"),
        durability="sync",
    )
    left = await graph.aget_state(run_config("thread-leave"))
    assert interrupts_of(left)
    assert left.values["pending"] == "goal"
    assert left.values["question"] == "QQQ?"
    assert left.values["user_question"] == asked
    assert "Готовые книги:" not in left.values["user_question"]
    assert len(parser.catalog_calls) == searched
    _act(embedder, "unclear")
    await _run(graph, "RRR?", thread="thread-label")
    _score(ranker, "r0")
    await graph.ainvoke(
        Command(resume="CAPEX"),
        run_config("thread-label"),
        durability="sync",
    )
    labeled = await graph.aget_state(run_config("thread-label"))
    assert [call["q"] for call in parser.catalog_calls].count("CAPEX") == 0
    assert labeled.values["question"] == "RRR?"
    assert "5" in labeled.values["draft"]
    assert labeled.values["draft"] != introduction
    overview = book_overview(parser.context)
    about_before = len(_seen(model, "about"))
    catalog_before = len(parser.catalog_calls)
    _act(embedder, "overview")
    await graph.ainvoke(
        new_turn_input("packt-project-finance.xlsx"),
        run_config("thread-bare-file"),
        durability="sync",
    )
    opened = await graph.aget_state(run_config("thread-bare-file"))
    assert not interrupts_of(opened)
    assert opened.values["job_id"] == "job-1"
    assert parser.context_calls[-1] == "job-1"
    assert opened.values["draft"] == overview
    assert opened.values["draft"] != books_reply(parser.jobs)
    assert "Готовые книги:" not in opened.values["draft"]
    assert len(_seen(model, "about")) == about_before
    assert len(parser.catalog_calls) == catalog_before
    _act(embedder, "overview")
    await graph.ainvoke(
        new_turn_input("дай саммари по packt-project-finance.xlsx"),
        run_config("thread-summary"),
        durability="sync",
    )
    summary = await graph.aget_state(run_config("thread-summary"))
    assert summary.values["draft"] == overview
    assert summary.values["job_id"] == "job-1"
    assert "Готовые книги:" not in summary.values["draft"]
    assert len(parser.catalog_calls) == catalog_before
    _act(embedder, "overview")
    await graph.ainvoke(
        new_turn_input("что в книге packt-project-finance.xlsx ?"),
        run_config("thread-contents"),
        durability="sync",
    )
    contents = await graph.aget_state(run_config("thread-contents"))
    assert contents.values["draft"] == overview
    assert contents.values["job_id"] == "job-1"
    assert len(parser.catalog_calls) == catalog_before
    heard_books = len(parser.context_calls)
    _act(embedder, "files")
    await graph.ainvoke(
        new_turn_input("какие книги ?"),
        run_config("thread-which"),
        durability="sync",
    )
    which = await graph.aget_state(run_config("thread-which"))
    assert not interrupts_of(which)
    assert which.values["awaiting"] == ""
    assert which.values["draft"] == books_reply(parser.jobs)
    assert len(parser.context_calls) == heard_books
    _score(ranker, "r0")
    await graph.ainvoke(
        new_turn_input("DSCR в packt-project-finance.xlsx"),
        run_config("thread-dscr-file"),
        durability="sync",
    )
    dscr = await graph.aget_state(run_config("thread-dscr-file"))
    assert parser.catalog_calls == []
    assert not any("packt" in call["q"].casefold() for call in parser.catalog_calls)
    assert "Готовые книги:" not in (dscr.values.get("draft") or "")
    assert dscr.values["satisfactory"] is True
    assert dscr.values["citations"][0]["row_key"] == "row-dscr"
    both_about = len(_seen(model, "about"))
    await graph.ainvoke(
        new_turn_input("packt-project-finance.xlsx и rvi-project-finance.xlsx"),
        run_config("thread-both"),
        durability="sync",
    )
    both = await graph.aget_state(run_config("thread-both"))
    assert interrupts_of(both)
    assert "Какую книгу открыть?" in both.values["user_question"]
    assert "packt-project-finance.xlsx" in both.values["user_question"]
    assert "rvi-project-finance.xlsx" in both.values["user_question"]
    assert both.values.get("draft") != books_reply(parser.jobs)
    assert len(_seen(model, "about")) == both_about
    _act(embedder, "greet")
    hello_context = len(parser.context_calls)
    await graph.ainvoke(
        new_turn_input("привет", "job-1"),
        run_config("thread-hello-slot"),
        durability="sync",
    )
    hello = await graph.aget_state(run_config("thread-hello-slot"))
    assert interrupts_of(hello)
    assert hello.values["pending"] == "goal"
    assert hello.values["job_id"] == "job-1"
    assert hello.values["user_question"] == asked
    assert "rvi-project-finance.xlsx" not in hello.values["user_question"]
    assert "Готовые книги:" not in hello.values["user_question"]
    assert len(parser.context_calls) == hello_context
    _act(embedder, "figure")
    await graph.ainvoke(
        new_turn_input("Какой DSCR?"),
        run_config("thread-pick"),
        durability="sync",
    )
    menu = await graph.aget_state(run_config("thread-pick"))
    assert interrupts_of(menu)
    assert "Какую книгу открыть?" in menu.values["user_question"]
    _score(ranker, "r0")
    await graph.ainvoke(
        Command(resume="открой packt-project-finance.xlsx"),
        run_config("thread-pick"),
        durability="sync",
    )
    picked = await graph.aget_state(run_config("thread-pick"))
    assert picked.values["job_id"] == "job-1"
    assert parser.catalog_calls == []
    assert picked.values["satisfactory"] is True
    assert picked.values["citations"][0]["row_key"] == "row-dscr"


def _book_document() -> dict:
    return {
        "meta": {"source_filename": "rvi-project-finance.xlsx"},
        "workbook": {
            "sheets": ["Cover Page", "Top Shortcuts", "PF Model"],
            "formula_count": 10219,
            "missing_cached_values": 20,
        },
        "warnings": ["cache gap"],
        "mapping_stats": {"mapped": 1, "concept_coverage": 0.5},
        "graph": {"artifact": "graph.json"},
        "blocks": [
            {
                "sheet": "PF Model",
                "rows": [
                    {
                        "label": "Inputs Time Dependent",
                        "kind": "abstract",
                        "disposition": "header",
                    },
                    {
                        "label": "CPI",
                        "kind": "fact",
                        "disposition": "mapped",
                        "concept_id": "secret-concept",
                        "hints": {"scale": "m"},
                        "numeric_summary": {
                            "constant": True,
                            "first": "0.02",
                            "last": "0.02",
                            "minimum": "0.02",
                            "maximum": "0.02",
                            "n": 3,
                        },
                        "cells": [{"role": "unit", "cached_value": "%", "addr": "B1"}],
                        "values": ["0.02", "0.05", "0.08"],
                    },
                    {
                        "label": "Ставка",
                        "kind": "fact",
                        "disposition": "mapped",
                        "numeric_summary": {
                            "constant": True,
                            "first": "7.927055656909944E-2",
                            "last": "7.927055656909944E-2",
                            "minimum": "7.927055656909944E-2",
                            "maximum": "7.927055656909944E-2",
                        },
                        "cells": [{"role": "unit", "cached_value": "%"}],
                    },
                    {
                        "label": "Поток",
                        "kind": "fact",
                        "disposition": "mapped",
                        "hints": {"scale": "k"},
                        "numeric_summary": {
                            "constant": False,
                            "first": "0",
                            "last": "0",
                            "minimum": "0",
                            "maximum": "8168.3204057963",
                        },
                        "cells": [
                            {"role": "unit", "cached_value": "EUR'000"},
                            {"role": "total", "cached_value": "99999"},
                        ],
                    },
                    {
                        "label": "Срок",
                        "kind": "fact",
                        "disposition": "mapped",
                        "numeric_summary": {
                            "constant": True,
                            "first": "1",
                            "last": "1",
                            "minimum": "1",
                            "maximum": "1",
                        },
                    },
                    {
                        "label": "Доля",
                        "kind": "fact",
                        "disposition": "mapped",
                        "numeric_summary": {
                            "constant": False,
                            "first": "0",
                            "last": "3.5545123789273241",
                            "minimum": "0",
                            "maximum": "3.5545123789",
                        },
                        "cells": [{"role": "unit", "cached_value": "%"}],
                    },
                    {
                        "label": "Годовых",
                        "kind": "fact",
                        "disposition": "mapped",
                        "numeric_summary": {
                            "constant": True,
                            "first": "3.5000000000000003E-2",
                            "last": "3.5000000000000003E-2",
                            "minimum": "0.035",
                            "maximum": "0.035",
                        },
                        "cells": [{"role": "unit", "cached_value": "% p.a."}],
                    },
                    {
                        "label": "Долг",
                        "kind": "fact",
                        "disposition": "mapped",
                        "hints": {"scale": "k"},
                        "cells": [
                            {"role": "unit", "cached_value": "EUR'000"},
                            {"role": "value", "cached_value": "60000"},
                            {
                                "role": "value",
                                "header": "Gearing",
                                "cached_value": "0.60060060060060061",
                            },
                        ],
                    },
                    {
                        "label": "Выбор",
                        "kind": "fact",
                        "disposition": "mapped",
                        "numeric_summary": {
                            "constant": False,
                            "first": "CPI",
                            "last": "CPI",
                        },
                        "cells": [{"role": "value", "header": "Live Case", "cached_value": "CPI"}],
                    },
                    {
                        "label": "Доходность",
                        "kind": "fact",
                        "disposition": "mapped",
                        "cells": [
                            {"role": "unit", "cached_value": "%"},
                            {
                                "role": "value",
                                "header": "IRR",
                                "cached_value": "7.927055656909944E-2",
                            },
                        ],
                    },
                    {
                        "label": "Ценность",
                        "kind": "fact",
                        "disposition": "mapped",
                        "hints": {"scale": "k"},
                        "cells": [
                            {"role": "unit", "cached_value": "EUR'000"},
                            {
                                "role": "value",
                                "header": "Target IRR",
                                "cached_value": "11470.633594198856",
                            },
                        ],
                    },
                    {
                        "label": "Пустая строка",
                        "kind": "fact",
                        "disposition": "abstained",
                    },
                    {
                        "label": "Checks",
                        "kind": "abstract",
                        "disposition": "header",
                    },
                    {
                        "label": "Spare",
                        "kind": "helper",
                        "disposition": "excluded",
                        "numeric_summary": {
                            "constant": True,
                            "first": "99",
                            "last": "99",
                            "minimum": "99",
                            "maximum": "99",
                        },
                    },
                ],
            }
        ],
    }


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Сравни Выручка и EBITDA в 2030")
    assert len(parser.observation_calls) == 1
    assert parser.observation_calls[0]["row_key"] == "row-rev"
    values = ranker.calls[-1]["criteria"].values()
    assert "Выручка" in values
    assert "EBITDA" in values
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    parser.observations[("job-1", "row-obs")] = [
        observation("row-obs", "2030", "1.25", "C10", label="DSCR наблюдённый")
    ]
    model = ScriptedModel()
    ranker = FakeRanker()
    _offer_both(ranker)
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой DSCR в 2030?")
    paused = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(paused)
    assert "999" not in paused.values["user_question"]
    assert "Какую строку" in paused.values["user_question"]
    await graph.ainvoke(
        Command(resume="1"),
        run_config("thread-1"),
        durability="sync",
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["row_key"] == "row-obs"
    assert snap.values["citations"][0]["cell"] == "C10"
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_why_after_a_finished_answer_does_not_ask_the_row_again() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    stored = observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    stored["formula"]["text"] = "=H10/I10"
    parser.observations[("job-1", "row-dscr")] = [stored]
    model = ScriptedModel()
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker, embedder=embedder)
    await _run(graph, "Какой DSCR в 2030?")
    searched = len(parser.catalog_calls)
    _act(embedder, "explain")
    await graph.ainvoke(
        new_turn_input("А почему?", "job-1"), run_config("thread-1"), durability="sync"
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True
    assert "=H10/I10" in snap.values["draft"]
    assert "1.25" in snap.values["draft"]
    assert _seen(model, "plan") == []
    assert _seen(model, "answer") == []
    assert len(parser.catalog_calls) == searched
    assert parser.observation_calls[-1]["precedent_depth"] == 2
    assert parser.observation_calls[-1]["row_key"] == "row-dscr"
    fetched = len(parser.observation_calls)
    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("Спасибо", "job-1"), run_config("thread-1"), durability="sync"
    )
    thanks = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(thanks)
    assert thanks.values["pending"] == "goal"
    assert thanks.values["user_question"] == goal_question("")
    assert "1.25" not in thanks.values["user_question"]
    assert "Готовые книги:" not in thanks.values["user_question"]
    assert len(parser.observation_calls) == fetched


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    _score(ranker, "r0")
    graph = _graph(parser, ByQuestion(), ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    model = ScriptedModel()
    ranker = FakeRanker()
    ranker.fail()
    graph = _graph(parser, model, ranker=ranker)
    with pytest.raises(ModelError):
        await _run(graph, "Какой DSCR?")
    assert parser.catalog_calls == []
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_truncated_series_asks_for_a_period() -> None:
    parser = FakeParser()
    parser.force_truncated = True
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    model.push("plan", plan(["DSCR"]))
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
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
    embedder = FakeEmbed()
    embedder.fail()
    model.push("talk", "boom")
    _act(embedder, "greet")
    graph = _graph(parser, model, embedder=embedder)
    with pytest.raises(ModelError):
        await _run(graph, "Какой?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.next
    assert not interrupts_of(snap)
    await graph.ainvoke(None, run_config("thread-1"), durability="sync")
    done = await graph.aget_state(run_config("thread-1"))
    assert done.values["question"] == "Какой?"
    assert interrupts_of(done)
    assert done.values["pending"] == "goal"
    assert done.values["user_question"] == goal_question("")
    assert done.values["job_id"] == "job-1"


@pytest.mark.asyncio
async def test_fourth_distinct_gap_stops_the_budget() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    parser.observations[("job-1", "row-dscr")] = [observation("row-dscr", "2030", "1.25", "C10")]
    model = ScriptedModel()
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой DSCR в 2030?")
    assert _seen(model, "answer") == []
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "1.25"
    assert snap.values["content_steps"] == 1


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой Operating Income or Loss (EBITDA) в 2026?")
    assert parser.catalog_calls == []
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["citations"][0]["row_key"] == "PF|262"
    assert snap.values["citations"][0]["value"] == "10"
    assert _seen(model, "plan") == []


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Какой Debt service в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert parser.observation_calls[0]["row_key"] == "Debt|17"
    assert [item["row_key"] for item in snap.values["citations"]] == ["Debt|17"]


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Какой EBITDA в первый операционный год?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.observation_calls == []
    assert interrupts_of(snap)
    assert "оси" in snap.values["user_question"]
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, f"Какой {label} в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.observation_calls == []
    assert interrupts_of(snap)
    assert "оси" in snap.values["user_question"]
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Какой DSCR в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert _seen(model, "answer") == []
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"][0]["value"] == "12.5"
    assert "тыс." in snap.values["draft"]
    assert snap.values["gaps"] == []


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Покажи CFADS, Total debt service и DSCR в 2030")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.catalog_calls == []
    assert [item["row_key"] for item in snap.values["citations"]] == ["debt"]
    assert {item["period_id"] for item in snap.values["citations"]} == {"2030"}
    assert "Total debt service" in ranker.calls[-1]["criteria"].values()
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
    parser.observations[("job-1", "a")] = [observation("a", "2030", "10", "A1", label="CFADS")]
    parser.observations[("job-1", "c")] = [observation("c", "2030", "20", "A2", label="EBITDA")]
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Сравни CFADS и EBITDA в 2030")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert len(ranker.calls) == 1
    assert parser.catalog_calls == []
    assert [item["row_key"] for item in snap.values["citations"]] == ["a"]
    assert [call["row_key"] for call in parser.observation_calls] == ["a"]


@pytest.mark.asyncio
async def test_five_needles_asks_to_narrow() -> None:
    labels = ("CFADS", "EBITDA", "DSCR", "IRR", "Debt")
    parser = FakeParser()
    parser.axes = year_axes()
    for index, label in enumerate(labels, start=1):
        key = f"row-{index}"
        parser.pages[("job-1", label.casefold())] = page(
            [catalog_row(key, label, axis=["forecast"])]
        )
        parser.observations[("job-1", key)] = [
            observation(key, "", str(index), f"A{index}", label=label)
        ]
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "CFADS, EBITDA, DSCR, IRR, Debt")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True
    assert [item["row_key"] for item in snap.values["citations"]] == ["row-1"]
    assert parser.catalog_calls == []
    assert "четырёх" not in (snap.values.get("draft") or "")
    tight = FakeParser()
    tight.axes = year_axes()
    for index, label in enumerate(labels, start=1):
        tight.pages[("job-1", label.casefold())] = page(
            [catalog_row(f"row-{index}", label, axis=["forecast"])]
        )
        tight.observations[("job-1", f"row-{index}")] = [
            observation(f"row-{index}", "", str(index), f"A{index}", label=label)
        ]
    tight_ranker = FakeRanker()
    _score(tight_ranker, "r0")
    tight_graph = _graph(tight, ScriptedModel(), Settings(observation_cap=3), ranker=tight_ranker)
    await _run(tight_graph, "CFADS, EBITDA, DSCR, IRR, Debt", thread="thread-cap")
    published = await tight_graph.aget_state(run_config("thread-cap"))
    assert not interrupts_of(published)
    assert published.values["satisfactory"] is True
    assert [item["row_key"] for item in published.values["citations"]] == ["row-1"]
    assert "Назовите меньше подписей" not in (published.values.get("draft") or "")
    assert "четырёх" not in (published.values.get("draft") or "")
    assert tight.catalog_calls == []


@pytest.mark.asyncio
async def test_cover_question_pauses_without_catalog() -> None:
    parser = FakeParser()
    parser.summary = {"sheets": ["P&L", "CFS", "Debt"], "marker": "passport"}
    parser.axes = year_axes()
    model = ScriptedModel()
    embedder = FakeEmbed()
    _act(embedder, "figure")
    graph = _graph(parser, model, embedder=embedder)
    await _run(graph, "Какая общая картина?")
    snap = await graph.aget_state(run_config("thread-1"))
    question = snap.values["user_question"]
    assert interrupts_of(snap)
    assert parser.catalog_calls == []
    assert "Такой строки нет" in question
    assert "CFADS" not in question
    assert "DSCR" not in question
    assert "IRR" not in question
    assert not any(char.isdigit() for char in question)
    assert _seen(model, "plan") == []
    assert _seen(model, "about") == []


@pytest.mark.asyncio
async def test_ungrounded_overview_publishes_the_book_passport() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.context = _book_document()
    question = "Расскажи по модели ! дай саммари по ней"
    model = ScriptedModel()
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "overview")
    _act(embedder, "overview")
    graph = _graph(parser, model, ranker=ranker, embedder=embedder)
    await _run(graph, question)
    snap = await graph.aget_state(run_config("thread-1"))
    draft = snap.values["draft"]
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True
    assert snap.values["gaps"] == []
    assert snap.values["citations"] == []
    for piece in (
        "rvi-project-finance.xlsx",
        "## Inputs Time Dependent",
        "- CPI — 2%, млн.",
        "- Ставка — 7.93%",
        "- Поток — до 8 168.32, тыс. EUR",
        "- Срок — 1",
        "- Доля — с 0 по 3.55, %",
        "- Годовых — 3.5% p.a.",
        "- Долг — 60 000, тыс. EUR, Gearing: 0.6",
        "- Выбор — Live Case: CPI",
        "- Доходность — IRR: 7.93%",
        "- Ценность — Target IRR: 11 470.63, тыс. EUR",
    ):
        assert piece in draft
    for piece in (
        "10219",
        "Формул",
        "Предупреждение",
        "Единица",
        "Назовите подписи",
        "secret-concept",
        "graph.json",
        "0.5",
        "0.02",
        "E-",
        "Spare",
        "99",
        "Checks",
        "Пустая строка",
        "с 0 по 0",
        "с 0%",
        "99 999",
        "EUR'000",
    ):
        assert piece not in draft
    assert _seen(model, "about") == []
    assert _seen(model, "plan") == []
    assert parser.catalog_calls == []
    assert parser.observation_calls == []
    assert parser.context_calls == ["job-1"]
    assert ranker.calls == []
    assert len(embedder.calls) == 1
    await _run(graph, question)
    again = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(again)
    assert again.values["satisfactory"] is True
    assert again.values["gaps"] == []
    assert again.values["draft"] == draft
    assert parser.catalog_calls == []
    assert parser.context_calls == ["job-1", "job-1"]
    assert len(embedder.calls) == 2


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
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, f"Ключевые показатели: {left} и {right}")
    snap = await graph.aget_state(run_config("thread-1"))
    assert [item["row_key"] for item in snap.values["citations"]] == ["avg"]
    values = ranker.calls[-1]["criteria"].values()
    assert left in values
    assert right in values
    assert parser.catalog_calls == []


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
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "figure")
    graph = _graph(parser, model, ranker=ranker, embedder=embedder)
    await _run(graph, "Какая общая картина?")
    _score(ranker, "r0")
    await graph.ainvoke(
        Command(resume="CFADS и EBITDA в 2030"),
        run_config("thread-1"),
        durability="sync",
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert [item["row_key"] for item in snap.values["citations"]] == ["cfs"]
    assert {item["period_id"] for item in snap.values["citations"]} == {"2030"}
    values = ranker.calls[-1]["criteria"].values()
    assert "CFADS" in values
    assert "EBITDA" in values
    assert _seen(model, "plan") == []


@pytest.mark.asyncio
async def test_parenthetical_label_publishes_the_cache_from_the_ranker_key() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    exact = "CAPEX (including SPV costs)"
    heavy = "CAPEX (including heavy maintenance & SPV costs)"
    parser.rows["job-1"] = [
        catalog_row(
            "exact",
            exact,
            sheet="Input Assumptions",
            label_path=["COSTS DURING CONSTRUCTION"],
        ),
        catalog_row("heavy", heavy, sheet="Construction"),
    ]
    parser.observations[("job-1", "exact")] = [
        observation("exact", "scenario-4", "300000000", "C4", label=exact),
        observation("exact", "base-case", "300000000", "C1", label=exact),
    ]
    ranker = FakeRanker()
    ranker.push("r0", {"r0": 0.7, "r1": 0.2, "books": 0.04, "book": 0.03, "intro": 0.02})
    ranker.push("r0", {"r0": 0.7, "r1": 0.2, "books": 0.04, "book": 0.03, "intro": 0.02})
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    question = "расскажи про CAPEX (including SPV costs) по годам"
    await _run(graph, question)
    snap = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == 1
    heard = ranker.calls[0]["criteria"]
    assert exact in heard.values()
    assert heavy in heard.values()
    assert parser.catalog_calls == []
    assert [row["row_key"] for row in snap.values["selected"]] == ["exact"]
    assert "300000000" in snap.values["draft"]
    assert "scenario-4" in snap.values["draft"]
    assert "Не хватает" not in snap.values["draft"]
    assert snap.values["satisfactory"] is True
    await graph.ainvoke(
        new_turn_input(question, "job-1"), run_config("thread-1"), durability="sync"
    )
    again = await graph.aget_state(run_config("thread-1"))
    assert again.values["draft"] == snap.values["draft"]
    assert len(ranker.calls) == 2


@pytest.mark.asyncio
async def test_two_contained_labels_are_both_selected() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = [
        catalog_row("e", "EBITDA"),
        catalog_row("d", "DSCR"),
    ]
    parser.observations[("job-1", "e")] = [observation("e", "2030", "10", "A1", label="EBITDA")]
    parser.observations[("job-1", "d")] = [observation("d", "2030", "1.2", "B1", label="DSCR")]
    ranker = FakeRanker()
    ranker.push("r0", {"r0": 0.8, "r1": 0.1, "books": 0.04, "book": 0.03, "intro": 0.02})
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Сравни EBITDA и DSCR")
    snap = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == 1
    heard = set(ranker.calls[0]["criteria"].values())
    assert "EBITDA" in heard
    assert "DSCR" in heard
    assert {item["row_key"] for item in snap.values["citations"]} == {"e"}


@pytest.mark.asyncio
async def test_short_label_menus_the_longer_line() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    long = "Operating Income or Loss (EBITDA)"
    parser.rows["job-1"] = [
        catalog_row("long", long, sheet="PF Model", label_path=["Income Statement"]),
        catalog_row("short", "EBITDA", sheet="PF Model", label_path=["Taxable income"]),
    ]
    parser.observations[("job-1", "long")] = [
        observation("long", "2030", "7694", "E185", label=long)
    ]
    ranker = FakeRanker()
    ranker.push("r0", {"r0": 0.85, "r1": 0.08, "books": 0.03, "book": 0.02, "intro": 0.01})
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Какой EBITDA в 2030?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == 1
    heard = set(ranker.calls[0]["criteria"].values())
    assert long in heard
    assert "EBITDA" in heard
    assert not interrupts_of(snap)
    assert "7694" in snap.values["draft"]
    assert "Какую строку" not in snap.values["draft"]
    split = FakeRanker()
    _offer_both(split)
    paused_graph = _graph(parser, ScriptedModel(), ranker=split)
    await _run(paused_graph, "Какой EBITDA в 2030?", thread="thread-menu")
    paused = await paused_graph.aget_state(run_config("thread-menu"))
    question = paused.values["user_question"]
    assert interrupts_of(paused)
    assert long in question
    assert "Лист PF Model" in question


@pytest.mark.asyncio
async def test_absent_label_stays_a_pause_when_the_book_action_is_weak() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = [catalog_row("e", "EBITDA")]
    parser.context = _book_document()
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "figure")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await _run(graph, "Какой DSCR?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert parser.context_calls == []
    assert ranker.calls == []


@pytest.mark.asyncio
async def test_ranker_picks_the_row_when_the_full_label_is_absent() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    label = "CAPEX (including SPV costs)"
    parser.rows["job-1"] = [catalog_row("long", label, sheet="Input Assumptions")]
    parser.observations[("job-1", "long")] = [
        observation("long", "2030", "300000000", "C1", label=label)
    ]
    ranker = FakeRanker()
    ranker.push("r0", {"r0": 0.8, "books": 0.05, "book": 0.04, "intro": 0.03})
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "расскажи про SPV costs по годам")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.catalog_calls == []
    assert "r0" in ranker.calls[0]["criteria"]
    assert snap.values["selected"][0]["row_key"] == "long"
    assert "300000000" in snap.values["draft"]


@pytest.mark.asyncio
async def test_open_menu_phrase_uses_the_ranker_number() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = [
        catalog_row("first", "CAPEX", sheet="Input Assumptions"),
        catalog_row("second", "CAPEX", sheet="Construction"),
    ]
    parser.observations[("job-1", "first")] = [
        observation("first", "2030", "300000000", "C1", label="CAPEX")
    ]
    ranker = FakeRanker()
    _offer_both(ranker)
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "CAPEX")
    paused = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(paused)
    assert len(ranker.calls) == 1
    heard = len(ranker.calls)
    await graph.ainvoke(Command(resume="1"), run_config("thread-1"), durability="sync")
    chosen = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == heard
    assert chosen.values["selected"][0]["row_key"] == "first"
    marked = FakeRanker()
    _offer_both(marked)
    marked_graph = _graph(parser, ScriptedModel(), ranker=marked)
    await _run(marked_graph, "CAPEX", thread="thread-mark")
    before_mark = len(marked.calls)
    await marked_graph.ainvoke(
        Command(resume="№1"), run_config("thread-mark"), durability="sync"
    )
    mark = await marked_graph.aget_state(run_config("thread-mark"))
    assert len(marked.calls) == before_mark
    assert mark.values["selected"][0]["row_key"] == "first"
    phrase = FakeRanker()
    phrase_embed = FakeEmbed()
    _offer_both(phrase)
    _act(phrase_embed, "figure")
    phrase_graph = _graph(parser, ScriptedModel(), ranker=phrase, embedder=phrase_embed)
    await _run(phrase_graph, "CAPEX", thread="thread-phrase")
    await phrase_graph.ainvoke(
        Command(resume="давай номер 1"),
        run_config("thread-phrase"),
        durability="sync",
    )
    snap = await phrase_graph.aget_state(run_config("thread-phrase"))
    assert len(phrase.calls) == 1
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert "300000000" not in (snap.values.get("draft") or "")


@pytest.mark.asyncio
async def test_open_menu_keeps_the_rows_when_the_ranker_is_unsure() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = [
        catalog_row("first", "CAPEX", sheet="Input Assumptions"),
        catalog_row("second", "CAPEX", sheet="Construction"),
    ]
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _offer_both(ranker)
    _act(embedder, "figure")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await _run(graph, "CAPEX")
    await graph.ainvoke(Command(resume="N1"), run_config("thread-1"), durability="sync")
    snap = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == 1
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert parser.observation_calls == []
    assert "300000000" not in snap.values.get("draft", "")


_PACKT = "packt-project-finance.xlsx"
_CACHED = "46482.849566458375"


def _debt_catalog() -> list[dict]:
    labels = (
        ("tba", "DEBT TIMELINE", "TBA"),
        ("src", "Debt (k£)", "Construction"),
        ("sched", "Debt Drawdown Schedule", "Construction"),
        ("bop", "Debt Outstanding BoP (k£)", "Construction"),
        ("draw", "Debt Drawdown (k£)", "Construction"),
        ("eop", "Debt Outstanding EoP (k£)", "Construction"),
        ("repay", "Debt Repayment Schedule", "Debt"),
        ("service", "Debt service", "Debt"),
        ("bs", "Debt", "Balance Sheet"),
    )
    return [catalog_row(key, label, sheet=sheet) for key, label, sheet in labels]


@pytest.mark.asyncio
async def test_summary_of_the_open_book_is_the_overview() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.summary = {"source_filename": _PACKT, "marker": "passport", "coverage": {"mapped": 1}}
    parser.context = _book_document()
    parser.rows["job-1"] = [
        catalog_row("cover", "PROJECT FINANCING", sheet="Input Assumptions"),
        catalog_row("irr-a", "Project IRR", sheet="Ratios"),
        catalog_row("irr-b", "Project IRR", sheet="Ratios"),
    ]
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "overview")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await _run(graph, f"сделай саммари {_PACKT}")
    snap = await graph.aget_state(run_config("thread-1"))
    assert ranker.calls == []
    assert embedder.calls
    assert "PROJECT FINANCING" not in embedder.calls[0]
    assert snap.values["draft"] == book_overview(parser.context)
    assert "Какую строку взять?" not in snap.values["draft"]
    assert snap.values["selected"] == []


@pytest.mark.asyncio
async def test_debt_outstanding_bop_publishes_the_cached_series() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = _debt_catalog()
    parser.observations[("job-1", "bop")] = [
        observation("bop", "Y3", _CACHED, "C3", label="Debt Outstanding BoP (k£)")
    ]
    ranker = FakeRanker()
    embedder = FakeEmbed()
    ranker.push(
        "r0",
        {"r0": 0.60, "r1": 0.39, "books": 0.02, "book": 0.02, "intro": 0.01},
    )
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    question = "Debt Outstanding BoP"
    await _run(graph, question)
    snap = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == 1
    values = list(ranker.calls[0]["criteria"].values())
    assert any("BoP" in item for item in values)
    assert any("EoP" in item for item in values)
    assert not any("Drawdown" in item or item in {"Debt", "Debt service"} for item in values)
    assert ranker.calls[0]["state"] == question
    assert "Книга " not in ranker.calls[0]["state"]
    assert _CACHED in snap.values["draft"]
    assert "Какую строку взять?" not in snap.values["draft"]
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_debt_phrase_on_an_open_menu_reads_the_new_row() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = [
        catalog_row("cover-a", "PROJECT FINANCING", sheet="Input Assumptions"),
        catalog_row("cover-b", "PROJECT FINANCING", sheet="Cover"),
        *_debt_catalog(),
    ]
    parser.observations[("job-1", "bop")] = [
        observation("bop", "Y3", _CACHED, "C3", label="Debt Outstanding BoP (k£)")
    ]
    ranker = FakeRanker()
    _offer_both(ranker)
    ranker.push(
        "r0",
        {"r0": 0.60, "r1": 0.39, "books": 0.02, "book": 0.02, "intro": 0.01},
    )
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "PROJECT FINANCING")
    paused = await graph.aget_state(run_config("thread-1"))
    assert interrupts_of(paused)
    assert "PROJECT FINANCING" in paused.values["user_question"]
    await graph.ainvoke(
        Command(resume="Debt Outstanding BoP (k£)"),
        run_config("thread-1"),
        durability="sync",
    )
    snap = await graph.aget_state(run_config("thread-1"))
    assert len(ranker.calls) == 2
    second = list(ranker.calls[1]["criteria"].values())
    assert any("BoP" in item for item in second)
    assert any("EoP" in item for item in second)
    assert snap.values["selected"][0]["row_key"] == "bop"
    assert _CACHED in snap.values["draft"]
    assert "Какую строку взять?" not in snap.values["draft"]


@pytest.mark.asyncio
async def test_unbound_books_question_lists_the_files() -> None:
    parser = FakeParser()
    parser.jobs = [
        {"job_id": "job-1", "source_filename": "packt-project-finance.xlsx"},
        {"job_id": "job-2", "source_filename": "rvi-project-finance.xlsx"},
    ]
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "files")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await graph.ainvoke(
        new_turn_input("какие книги у тебя есть"), run_config("thread-books"), durability="sync"
    )
    snap = await graph.aget_state(run_config("thread-books"))
    assert ranker.calls == []
    assert embedder.calls
    assert snap.values["draft"].startswith("Готовые книги:")
    assert "packt-project-finance.xlsx" in snap.values["draft"]
    assert "rvi-project-finance.xlsx" in snap.values["draft"]
    assert not interrupts_of(snap)


@pytest.mark.asyncio
async def test_unbound_greeting_asks_which_book() -> None:
    parser = FakeParser()
    parser.jobs = [
        {"job_id": "job-1", "source_filename": "packt-project-finance.xlsx"},
        {"job_id": "job-2", "source_filename": "rvi-project-finance.xlsx"},
    ]
    parser.context = _book_document()
    embedder = FakeEmbed()
    _act(embedder, "greet")
    graph = _graph(parser, ScriptedModel(), embedder=embedder)
    await graph.ainvoke(new_turn_input("привет"), run_config("thread-hi"), durability="sync")
    snap = await graph.aget_state(run_config("thread-hi"))
    assert interrupts_of(snap)
    assert "Какую книгу открыть?" in snap.values["user_question"]
    assert "finance-context-agent" not in snap.values["user_question"]
    assert parser.context_calls == []


@pytest.mark.asyncio
async def test_filename_after_a_greeting_is_the_introduction() -> None:
    parser = FakeParser()
    parser.jobs = [
        {"job_id": "job-1", "source_filename": _PACKT},
        {"job_id": "job-2", "source_filename": "rvi-project-finance.xlsx"},
    ]
    parser.summary = {"source_filename": _PACKT}
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "greet")
    _act(embedder, "greet")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await graph.ainvoke(new_turn_input("привет"), run_config("thread-hi"), durability="sync")
    paused = await graph.aget_state(run_config("thread-hi"))
    assert interrupts_of(paused)
    assert "Какую книгу открыть?" in paused.values["user_question"]
    assert ranker.calls == []
    await graph.ainvoke(Command(resume=_PACKT), run_config("thread-hi"), durability="sync")
    snap = await graph.aget_state(run_config("thread-hi"))
    assert interrupts_of(snap)
    assert snap.values["pending"] == "goal"
    assert snap.values["job_id"] == "job-1"
    assert snap.values["user_question"] == goal_question(_PACKT)
    assert "Что посмотреть?" in snap.values["user_question"]
    assert "Готовые книги:" not in snap.values["user_question"]
    assert "Готовые книги:" not in (snap.values.get("draft") or "")
    assert "Такой строки нет" not in snap.values["user_question"]
    assert parser.observation_calls == []
    assert parser.context_calls == []
    assert ranker.calls == []
    assert len(embedder.calls) == 2


@pytest.mark.asyncio
async def test_filename_after_a_missing_metric_stays_a_miss() -> None:
    parser = FakeParser()
    parser.jobs = [{"job_id": "job-1", "source_filename": _PACKT}]
    parser.summary = {"source_filename": _PACKT}
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "figure")
    _act(embedder, "figure")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await graph.ainvoke(
        new_turn_input("Какой DSCR?"), run_config("thread-dscr"), durability="sync"
    )
    paused = await graph.aget_state(run_config("thread-dscr"))
    assert interrupts_of(paused)
    assert "Какую книгу открыть?" in paused.values["user_question"]
    await graph.ainvoke(Command(resume=_PACKT), run_config("thread-dscr"), durability="sync")
    snap = await graph.aget_state(run_config("thread-dscr"))
    assert interrupts_of(snap)
    assert "Такой строки нет" in snap.values["user_question"]
    assert parser.context_calls == []
    assert snap.values["job_id"] == "job-1"
    assert ranker.calls == []
    assert len(embedder.calls) == 2


@pytest.mark.asyncio
async def test_greeting_inside_an_open_book_survives_a_miss() -> None:
    parser = FakeParser()
    parser.jobs = [{"job_id": "job-1", "source_filename": _PACKT}]
    parser.summary = {"source_filename": _PACKT}
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _act(embedder, "greet")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await graph.ainvoke(
        new_turn_input("Привет", "job-1"),
        run_config("thread-open-hi"),
        durability="sync",
    )
    snap = await graph.aget_state(run_config("thread-open-hi"))
    assert interrupts_of(snap)
    assert snap.values["pending"] == "goal"
    assert snap.values["user_question"] == goal_question(_PACKT)
    assert _PACKT in snap.values["user_question"]
    assert "Готовые книги:" not in snap.values["user_question"]
    assert snap.values["job_id"] == "job-1"
    assert ranker.calls == []
    assert parser.context_calls == []


def _capex_catalog() -> list[dict]:
    return [
        catalog_row("exact", "CAPEX (including SPV costs)", sheet="Input Assumptions"),
        catalog_row("bare", "CAPEX", sheet="Construction"),
        catalog_row("short", "CAPEX (incl. SPV costs)", sheet="Ratios"),
    ]


_RVI = "rvi-project-finance.xlsx"


@pytest.mark.asyncio
async def test_open_book_asks_what_to_look_at() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.jobs = [
        {"job_id": "job-1", "source_filename": _PACKT},
        {"job_id": "job-2", "source_filename": _RVI},
    ]
    parser.context = _book_document()
    parser.rows["job-1"] = _capex_catalog() + [catalog_row("ebitda", "EBITDA", sheet="P&L")]
    parser.observations[("job-1", "exact")] = [
        observation("exact", "Y1", "300000000", "C5", label="CAPEX (including SPV costs)")
    ]
    parser.observations[("job-1", "ebitda")] = [
        observation("ebitda", "Y1", "88", "C8", label="EBITDA")
    ]

    async def summaries(job_id: str):
        name = _PACKT if job_id == "job-1" else _RVI
        parser.summary = {"source_filename": name, "marker": "passport", "coverage": {"mapped": 1}}
        return parser.summary, parser.summary_etag

    parser.get_summary = summaries
    ranker = FakeRanker()
    embedder = FakeEmbed()
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    _act(embedder, "files")
    await graph.ainvoke(
        new_turn_input(f"открой {_PACKT}", "job-1"),
        run_config("thread-words"),
        durability="sync",
    )
    words = await graph.aget_state(run_config("thread-words"))
    assert interrupts_of(words)
    assert words.values["user_question"] == goal_question(_PACKT)
    assert "Готовые книги:" not in words.values["user_question"]
    assert ranker.calls == []
    await graph.ainvoke(
        Command(resume="Обзор книги"), run_config("thread-words"), durability="sync"
    )
    named = await graph.aget_state(run_config("thread-words"))
    assert not interrupts_of(named)
    assert named.values["draft"] == book_overview(parser.context)
    assert named.values["last_act"] == "overview"
    assert named.values["pending"] == ""
    assert named.values["menu_question"] == ""
    assert named.values["offers"] == []
    assert parser.observation_calls == []

    _act(embedder, "files")
    await graph.ainvoke(
        new_turn_input(f"открой {_PACKT}", "job-1"),
        run_config("thread-goal"),
        durability="sync",
    )
    opened = await graph.aget_state(run_config("thread-goal"))
    assert interrupts_of(opened)
    assert opened.values["pending"] == "goal"
    assert opened.values["user_question"] == goal_question(_PACKT)
    assert "Готовые книги:" not in (opened.values.get("draft") or "")
    assert ranker.calls == []
    await graph.ainvoke(Command(resume="1"), run_config("thread-goal"), durability="sync")
    overview = await graph.aget_state(run_config("thread-goal"))
    assert not interrupts_of(overview)
    assert overview.values["draft"] == book_overview(parser.context)
    assert overview.values["last_act"] == "overview"
    assert overview.values["pending"] == ""
    assert overview.values["offers"] == []

    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("привет", "job-1"), run_config("thread-goal"), durability="sync"
    )
    await graph.ainvoke(Command(resume="2"), run_config("thread-goal"), durability="sync")
    listed = await graph.aget_state(run_config("thread-goal"))
    assert not interrupts_of(listed)
    assert listed.values["last_act"] == "catalog"
    assert "Лист Input Assumptions" in listed.values["draft"]
    assert "Лист Construction" in listed.values["draft"]
    assert "| Атрибут | Раздел |" in listed.values["draft"]
    assert "300000000" not in listed.values["draft"]
    assert "88" not in listed.values["draft"]
    assert parser.observation_calls == []

    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("привет", "job-1"), run_config("thread-goal"), durability="sync"
    )
    await graph.ainvoke(Command(resume="3"), run_config("thread-goal"), durability="sync")
    label = await graph.aget_state(run_config("thread-goal"))
    assert interrupts_of(label)
    assert label.values["user_question"] == "Назовите подпись."
    embedded = len(embedder.calls)
    ranker.push("r0", {"r0": 0.9, "r1": 0.05}, sheets=0.10)
    await graph.ainvoke(
        Command(resume="CAPEX (including SPV costs)"),
        run_config("thread-goal"),
        durability="sync",
    )
    cited = await graph.aget_state(run_config("thread-goal"))
    assert not interrupts_of(cited)
    assert cited.values["citations"][0]["value"] == "300000000"
    assert cited.values["citations"][0]["row_key"] == "exact"
    assert len(embedder.calls) == embedded
    assert ranker.calls

    _act(embedder, "greet")
    await graph.ainvoke(
        new_turn_input("привет", "job-1"), run_config("thread-goal"), durability="sync"
    )
    await graph.ainvoke(Command(resume="4"), run_config("thread-goal"), durability="sync")
    other = await graph.aget_state(run_config("thread-goal"))
    assert interrupts_of(other)
    assert other.values["pending"] == "job"
    assert other.values["question"] == ""
    assert "Какую книгу открыть?" in other.values["user_question"]
    assert _PACKT in other.values["user_question"]
    assert _RVI in other.values["user_question"]
    before_switch = len(embedder.calls)
    await graph.ainvoke(
        Command(resume=_RVI), run_config("thread-goal"), durability="sync"
    )
    switched = await graph.aget_state(run_config("thread-goal"))
    assert switched.values["job_id"] == "job-2"
    assert interrupts_of(switched)
    assert switched.values["pending"] == "goal"
    assert switched.values["user_question"] == goal_question(_RVI)
    assert "Готовые книги:" not in switched.values["user_question"]
    assert len(embedder.calls) == before_switch

    _act(embedder, "files")
    await graph.ainvoke(
        new_turn_input(f"открой {_PACKT}", "job-1"),
        run_config("thread-metric"),
        durability="sync",
    )
    metric_embedded = len(embedder.calls)
    ranker.push("r0", {"r0": 0.9}, sheets=0.10)
    await graph.ainvoke(
        Command(resume="Какой EBITDA в Y1?"),
        run_config("thread-metric"),
        durability="sync",
    )
    metric = await graph.aget_state(run_config("thread-metric"))
    assert not interrupts_of(metric)
    assert metric.values["pending"] == ""
    assert metric.values["citations"][0]["value"] == "88"
    assert metric.values["citations"][0]["period_id"] == "Y1"
    assert len(embedder.calls) == metric_embedded
    assert len(ranker.calls) >= 2

    _act(embedder, "files")
    await graph.ainvoke(
        new_turn_input("какие книги ?"), run_config("thread-list"), durability="sync"
    )
    inventory = await graph.aget_state(run_config("thread-list"))
    assert not interrupts_of(inventory)
    assert inventory.values["draft"].startswith("Готовые книги:")
    assert _PACKT in inventory.values["draft"]
    assert _RVI in inventory.values["draft"]


@pytest.mark.asyncio
async def test_sheet_question_lists_every_capex() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = _capex_catalog()
    parser.observations[("job-1", "short")] = [
        observation("short", "Y1", "-75000", "C1", label="CAPEX (incl. SPV costs)"),
        observation("short", "Y5", "0", "C5", label="CAPEX (incl. SPV costs)"),
    ]
    ranker = FakeRanker()
    embedder = FakeEmbed()
    ranker.push("r2", {"r2": 0.71, "r0": 0.19, "r1": 0.10}, sheets=0.90)
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await _run(graph, "какие еще есть CAPEX в книге и на каких листах?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert not interrupts_of(snap)
    assert snap.values["satisfactory"] is True
    assert snap.values["citations"] == []
    assert snap.values["selected"] == []
    assert parser.observation_calls == []
    assert snap.values["draft"] == (
        "CAPEX (including SPV costs)\n"
        "Лист Input Assumptions.\n"
        "CAPEX\n"
        "Лист Construction.\n"
        "CAPEX (incl. SPV costs)\n"
        "Лист Ratios."
    )
    assert "-75000" not in snap.values["draft"]
    assert "Какую строку взять?" not in snap.values["draft"]
    assert "№1" not in snap.values["draft"]
    assert len(ranker.calls) == 1
    assert ranker.calls[0]["ask_act"] is True
    assert "sheets" not in ranker.calls[0]["criteria"]
    assert set(ranker.calls[0]["criteria"].values()) == {
        "CAPEX (including SPV costs)",
        "CAPEX",
        "CAPEX (incl. SPV costs)",
    }
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_low_sheets_score_still_prints_the_row() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = _capex_catalog()
    parser.observations[("job-1", "short")] = [
        observation("short", "Y1", "-75000", "C1", label="CAPEX (incl. SPV costs)")
    ]
    ranker = FakeRanker()
    ranker.push("r0", {"r0": 0.8, "r1": 0.1}, sheets=0.10)
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "CAPEX (incl. SPV costs)")
    snap = await graph.aget_state(run_config("thread-1"))
    assert "-75000" in snap.values["draft"]
    assert parser.observation_calls
    assert "Input Assumptions" not in snap.values["draft"]
    assert "Construction" not in snap.values["draft"]
    assert snap.values["citations"]


@pytest.mark.asyncio
async def test_ebitda_value_is_not_a_sheet_list() -> None:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.rows["job-1"] = [catalog_row("ebitda", "EBITDA", sheet="P&L")]
    parser.observations[("job-1", "ebitda")] = [
        observation("ebitda", "Y1", "88", "C1", label="EBITDA")
    ]
    ranker = FakeRanker()
    ranker.push("r0", {"r0": 0.9}, sheets=0.61)
    graph = _graph(parser, ScriptedModel(), ranker=ranker)
    await _run(graph, "Какой EBITDA в Y1?")
    snap = await graph.aget_state(run_config("thread-1"))
    assert parser.observation_calls
    assert snap.values["citations"]
    assert snap.values["citations"][0]["value"] == "88"
    assert snap.values["citations"][0]["period_id"] == "Y1"


def _attribute_book() -> list[dict]:
    return [
        catalog_row(
            "exact",
            "CAPEX (including SPV costs)",
            sheet="Input Assumptions",
            label_path=["COSTS DURING CONSTRUCTION"],
        ),
        catalog_row("bare", "CAPEX", sheet="Construction"),
        catalog_row(
            "short",
            "CAPEX (incl. SPV costs)",
            sheet="Ratios",
            label_path=["Project IRR"],
        ),
        catalog_row("ebitda", "EBITDA", sheet="P&L"),
    ]


def _attribute_parser() -> FakeParser:
    parser = FakeParser()
    parser.axes = year_axes()
    parser.summary = {"source_filename": _PACKT, "marker": "passport", "coverage": {"mapped": 1}}
    parser.rows["job-1"] = _attribute_book()
    return parser


_CAPEX_CATALOG = (
    "Лист Input Assumptions\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX (including SPV costs) | COSTS DURING CONSTRUCTION |\n"
    "\n"
    "Лист Construction\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX | |\n"
    "\n"
    "Лист Ratios\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX (incl. SPV costs) | Project IRR |\n"
    "\n"
    "Лист P&L\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| EBITDA | |"
)

_CONSTRUCTION_CATALOG = (
    "Лист Construction\n"
    "\n"
    "| Атрибут | Раздел |\n"
    "| --- | --- |\n"
    "| CAPEX | |"
)


def _below_gap(leader: str) -> dict[str, float]:
    """Leader clears the floor and stays a thousandth under the required gap."""
    second = ACT_FLOOR - ACT_GAP + 0.001
    scores = {name: second for name in ACTS}
    scores[leader] = ACT_FLOOR
    return scores


@pytest.mark.asyncio
async def test_attribute_questions_print_the_open_book_catalog() -> None:
    parser = _attribute_parser()
    for question in ("дай список атрибутов в этой книге", "какие атрибуты есть ?"):
        ranker = FakeRanker()
        embedder = FakeEmbed()
        model = ScriptedModel()
        _act(embedder, "catalog")
        graph = _graph(parser, model, ranker=ranker, embedder=embedder)
        await _run(graph, question, thread=f"thread-{len(question)}")
        snap = await graph.aget_state(run_config(f"thread-{len(question)}"))
        assert not interrupts_of(snap)
        assert snap.values["draft"] == _CAPEX_CATALOG
        assert snap.values["satisfactory"] is True
        assert snap.values["citations"] == []
        assert snap.values["selected"] == []
        assert snap.values["last_act"] == "catalog"
        assert ranker.calls == []
        assert model.seen == []
        assert parser.observation_calls == []
        assert "Такой строки нет" not in snap.values["draft"]
        assert "Готовые книги:" not in snap.values["draft"]
        assert not any(char.isdigit() for char in snap.values["draft"])


@pytest.mark.asyncio
async def test_named_sheet_keeps_only_that_sheet() -> None:
    parser = _attribute_parser()
    embedder = FakeEmbed()
    _act(embedder, "catalog")
    _act(embedder, "catalog")
    graph = _graph(parser, ScriptedModel(), embedder=embedder)
    await _run(graph, "что есть на листе Construction")
    direct = await graph.aget_state(run_config("thread-1"))
    assert direct.values["draft"] == _CONSTRUCTION_CATALOG
    assert "EBITDA" not in direct.values["draft"]
    assert "Input Assumptions" not in direct.values["draft"]
    await graph.ainvoke(
        new_turn_input("а на Construction?", "job-1"),
        run_config("thread-1"),
        durability="sync",
    )
    follow = await graph.aget_state(run_config("thread-1"))
    assert follow.values["draft"] == _CONSTRUCTION_CATALOG
    assert follow.values["last_act"] == "catalog"
    assert len(embedder.calls) == 2
    assert "прошлый ход: catalog" in embedder.calls[1]


@pytest.mark.asyncio
async def test_thin_gap_accepts_chat_only_when_it_names_the_leader() -> None:
    parser = _attribute_parser()
    agreed = FakeEmbed()
    agreed.push(_below_gap("catalog"))
    model = ScriptedModel()
    model.push("talk", {"act": "catalog"})
    graph = _graph(parser, model, embedder=agreed)
    await _run(graph, "список метрик")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["draft"] == _CAPEX_CATALOG
    assert _seen(model, "talk")
    split = FakeEmbed()
    split.push(_below_gap("catalog"))
    other = ScriptedModel()
    other.push("talk", {"act": "files"})
    other_graph = _graph(parser, other, embedder=split)
    await _run(other_graph, "список метрик", thread="thread-split")
    paused = await other_graph.aget_state(run_config("thread-split"))
    assert interrupts_of(paused)
    assert "Уточните:" in paused.values["user_question"]
    assert "Готовые книги:" not in (paused.values.get("draft") or "")
    assert "Готовые книги:" not in paused.values["user_question"]


@pytest.mark.asyncio
async def test_embed_failure_uses_the_chat_act() -> None:
    parser = _attribute_parser()
    embedder = FakeEmbed()
    embedder.fail()
    model = ScriptedModel()
    model.push("talk", {"act": "catalog"})
    graph = _graph(parser, model, embedder=embedder)
    await _run(graph, "дай список атрибутов в этой книге")
    snap = await graph.aget_state(run_config("thread-1"))
    assert snap.values["draft"] == _CAPEX_CATALOG
    assert snap.values["last_act"] == "catalog"
    assert _seen(model, "talk")


@pytest.mark.asyncio
async def test_explain_follows_the_open_row_and_a_bare_period_does_not_embed() -> None:
    parser = FakeParser()
    _bind(parser, "DSCR", [catalog_row("row-dscr", "DSCR")])
    stored = observation("row-dscr", "2030", "1.25", "C10", label="DSCR")
    stored["formula"]["text"] = "=H10/I10"
    parser.observations[("job-1", "row-dscr")] = [stored]
    ranker = FakeRanker()
    embedder = FakeEmbed()
    _score(ranker, "r0")
    _act(embedder, "explain")
    graph = _graph(parser, ScriptedModel(), ranker=ranker, embedder=embedder)
    await _run(graph, "Какой DSCR в 2030?")
    await graph.ainvoke(
        new_turn_input("как считается", "job-1"),
        run_config("thread-1"),
        durability="sync",
    )
    explained = await graph.aget_state(run_config("thread-1"))
    assert explained.values["satisfactory"] is True
    assert explained.values["draft"].startswith("DSCR в 2030: 1.25\nФормула C10: =H10/I10")
    assert "| Период |" not in explained.values["draft"]
    assert explained.values["last_act"] == "explain"
    assert "Лист " not in explained.values["draft"]
    assert parser.observation_calls[-1]["precedent_depth"] == 2
    assert len(embedder.calls) == 1
    heard = len(embedder.calls)
    await graph.ainvoke(
        new_turn_input("2030", "job-1"),
        run_config("thread-1"),
        durability="sync",
    )
    period = await graph.aget_state(run_config("thread-1"))
    assert period.values["satisfactory"] is True
    assert [(item["period_id"], item["value"]) for item in period.values["citations"]] == [
        ("2030", "1.25"),
    ]
    assert len(embedder.calls) == heard
    assert len(ranker.calls) == 1


_CONCESSION = "Concession Duration"
_CONCESSION_TABLE = (
    "| Период | Concession Duration |\n"
    "| --- | --- |\n"
    "| | 40 |\n"
    "| base-case | 40 |\n"
    "| scenario-2 | 36.5 |\n"
    "| gearing | 38 |\n"
    "| scenario-4 | 35.5 |"
)
_CONCESSION_SENTENCES = "\n".join(
    [
        "Concession Duration: 40",
        "Concession Duration в base-case: 40",
        "Concession Duration в scenario-2: 36.5",
        "Concession Duration в gearing: 38",
        "Concession Duration в scenario-4: 35.5",
    ]
)


def _concession_parser() -> FakeParser:
    parser = FakeParser()
    parser.axes = [{"id": "scenarios", "periods": []}]
    parser.pages[("job-1", _CONCESSION.casefold())] = page(
        [catalog_row("in|duration", _CONCESSION, axis=["scenarios"])]
    )
    parser.observations[("job-1", "in|duration")] = [
        observation("in|duration", "value", "40", "B1", label=_CONCESSION),
        observation("in|duration", "base-case", "40", "C1", label=_CONCESSION),
        observation("in|duration", "scenario-2", "36.5", "C2", label=_CONCESSION),
        observation("in|duration", "gearing", "38", "C3", label=_CONCESSION),
        observation("in|duration", "scenario-4", "35.5", "C4", label=_CONCESSION),
    ]
    return parser


async def _concession(model: ScriptedModel) -> dict:
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(_concession_parser(), model, ranker=ranker)
    await _run(graph, _CONCESSION)
    snap = await graph.aget_state(run_config("thread-1"))
    return snap.values


def _view_options(model: ScriptedModel) -> list[str]:
    seen = _seen(model, "view")
    assert len(seen) == 1
    return json.loads(seen[0]["options"])


@pytest.mark.asyncio
async def test_table_period_lays_concession_duration_out_by_period() -> None:
    model = ScriptedModel()
    model.push("view", {"view": "table-period"})
    values = await _concession(model)
    assert values["draft"] == _CONCESSION_TABLE
    assert "Concession Duration в base-case:" not in values["draft"]
    assert "в value" not in values["draft"]
    assert [item["period_id"] for item in values["citations"]] == [
        "",
        "base-case",
        "scenario-2",
        "gearing",
        "scenario-4",
    ]
    assert [item["value"] for item in values["citations"]] == ["40", "40", "36.5", "38", "35.5"]
    assert _view_options(model) == ["sentence", "table-period"]
    summary = json.loads(_seen(model, "view")[0]["user"])
    assert set(summary) == {
        "question",
        "points",
        "labels",
        "periods",
        "statuses",
        "scales",
        "explain",
    }
    assert summary["points"] == 5
    assert "36.5" not in _seen(model, "view")[0]["user"]
    assert "concept_id" not in _seen(model, "view")[0]["user"]


@pytest.mark.asyncio
async def test_a_chosen_sentence_stays_a_sentence() -> None:
    model = ScriptedModel()
    model.push("view", {"view": "sentence"})
    values = await _concession(model)
    assert values["draft"] == _CONCESSION_SENTENCES
    assert "| Период |" not in values["draft"]


@pytest.mark.asyncio
async def test_a_repeated_pair_offers_only_a_sentence() -> None:
    parser = _concession_parser()
    parser.observations[("job-1", "in|duration")] = [
        observation("in|duration", "base-case", "40", "C1", label=_CONCESSION),
        observation("in|duration", "base-case", "41", "C9", label=_CONCESSION),
    ]
    model = ScriptedModel()
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, _CONCESSION)
    snap = await graph.aget_state(run_config("thread-1"))
    assert _view_options(model) == ["sentence"]
    assert "| Период |" not in snap.values["draft"]


@pytest.mark.asyncio
async def test_a_broken_view_still_prints_the_period_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="finance_context_agent.graph")
    model = ScriptedModel()
    model.push("view", "bad")
    model.push("view", "bad")
    values = await _concession(model)
    assert values["draft"] == _CONCESSION_TABLE
    assert values["satisfactory"] is True
    messages = [record.message for record in caplog.records]
    assert "view fallback" in messages
    assert "view table-period" not in messages
    assert model.queues["view"] == ["bad"]


@pytest.mark.asyncio
async def test_a_dead_view_is_not_an_upstream_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="finance_context_agent.graph")
    model = ScriptedModel()
    model.push("view", "boom")
    values = await _concession(model)
    assert values["draft"] == _CONCESSION_TABLE
    assert "upstream_unavailable" not in values["draft"]
    assert "view fallback" in [record.message for record in caplog.records]


@pytest.mark.asyncio
async def test_two_explained_points_keep_formulas_under_the_table() -> None:
    parser = FakeParser()
    parser.axes = [
        {"id": "forecast", "periods": [{"period_key": "Y1"}, {"period_key": "Y2"}]}
    ]
    parser.pages[("job-1", "dscr")] = page(
        [catalog_row("row-dscr", "DSCR", axis=["forecast"])]
    )
    first = observation("row-dscr", "Y1", "1.10", "C1", label="DSCR")
    first["formula"] = {
        "text": "=H10/I10",
        "precedents": [{"cell": "H10", "label": "CFADS", "value": "15", "depth": 1}],
    }
    second = observation("row-dscr", "Y2", "1.25", "C2", label="DSCR")
    second["formula"] = {
        "text": "=I10/J10",
        "precedents": [{"cell": "I10", "label": "CFADS", "value": "20", "depth": 1}],
    }
    parser.observations[("job-1", "row-dscr")] = [first, second]
    model = ScriptedModel()
    model.push("view", {"view": "table-period"})
    ranker = FakeRanker()
    _score(ranker, "r0")
    graph = _graph(parser, model, ranker=ranker)
    await _run(graph, "Как считается DSCR")
    snap = await graph.aget_state(run_config("thread-1"))
    head, tail = snap.values["draft"].split("\n\n", 1)
    assert head.startswith("| Период | DSCR |")
    assert "| Y1 | 1.10 |" in head
    assert "| Y2 | 1.25 |" in head
    assert tail.startswith("Y1\n")
    assert "=H10/I10" in tail
    assert "=I10/J10" in tail
    assert "\n\n" not in tail
