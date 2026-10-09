"""Fixed LangGraph. Code names needles from the question and fetches slices."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from finance_context_agent.catalog import (
    MENU_INSTRUCTION,
    _heading,
    compact_row,
    matching_rows,
    narrow_page,
    offered_labels,
    render_menu,
)
from finance_context_agent.citations import verify_answer, wants_influence
from finance_context_agent.clients.llm import JsonModel, ModelError, SchemaError
from finance_context_agent.clients.parser import ParserClient, ParserError
from finance_context_agent.draft import cache_answer, scalar_observation
from finance_context_agent.periods import periods_from_question, resolve_periods
from finance_context_agent.prompts import (
    about_messages,
    answer_messages,
    talk_messages,
    without_account_code,
)
from finance_context_agent.questions import (
    ACTS,
    GOAL_CHOICES,
    HOLD_FLOOR,
    LIST_FLOOR,
    act_text,
    asks_how,
    book_overview,
    books_reply,
    catalog_reply,
    chitchat_reply,
    choice_rows,
    decide_margin,
    file_segments,
    goal_question,
    inventory_rows,
    match_goal,
    name_tokens,
    prototype_decision,
    ranker_state,
    sole_period_key,
)
from finance_context_agent.settings import Settings

_TOKEN_EDGE = "?.!,;:«»\"'[]"
_MISSING_ROW = "Такой строки нет. Назовите подпись иначе или выберите другую метрику."
_MENU_LEAD = "Какую строку взять?"
_MENU_NEEDLE = re.compile(r"«([^»]+)»\s*:")
_ASK_BOOKS = "Спросить, какие книги есть."
_ASK_BOOK = "Попросить обзор открытой книги."
_ASK_INTRO = "Реплика про самого агента."
_ASK_NONE = "Подходящей строки нет."
_CLARIFY = "Уточните: файлы, обзор книги, перечень атрибутов или одну метрику."
_EXPLAIN_ASK = "Назовите подпись, формулу которой объяснить."

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    job_id: str
    jobs: list[dict[str, Any]]
    pending: str
    awaiting: str
    user_question: str
    reload: bool
    just_bound_job: bool
    just_bound_row: bool
    offers: list[dict[str, Any]]
    terminal: str
    summary: dict[str, Any]
    summary_etag: str
    axes: list[dict[str, Any]]
    book_etag: str
    citations: list[dict[str, Any]]
    draft: str
    selected: list[dict[str, Any]]
    content_steps: int
    plan: dict[str, Any]
    human_reply: str
    search_again: bool
    gaps: list[str]
    gap_keys: list[str]
    same_gap: bool
    satisfactory: bool
    proposed_citations: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    period_ids: list[str]
    search_misses: int
    search_note: str
    last_user_question: str
    menu_question: str
    label_reply: bool
    clarify_rounds: int
    schema_error: bool
    draft_from_cache: bool
    cache_missing: bool
    last_act: str


def build_graph(
    parser: ParserClient,
    model: JsonModel,
    checkpointer: Any,
    settings: Settings | None = None,
    ranker: Any = None,
    embedder: Any = None,
) -> Any:
    settings = settings or Settings()
    label_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    async def load(state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("job_id"):
            return {
                "reload": False,
                "just_bound_job": False,
                "awaiting": "",
                "terminal": "",
            }
        try:
            summary, summary_etag = await parser.get_summary(state["job_id"])
            catalog, book_etag = await parser.get_axes(state["job_id"])
        except ParserError as exc:
            return _terminal_or_raise(exc)
        patch: dict[str, Any] = {
            "summary": summary,
            "summary_etag": summary_etag,
            "axes": catalog.get("axes") or [],
            "book_etag": book_etag,
            "reload": False,
            "just_bound_job": False,
            "awaiting": "",
            "terminal": "",
        }
        if state.get("book_etag") and state.get("book_etag") != book_etag:
            patch["citations"] = []
            patch["draft"] = ""
            patch["selected"] = []
        return patch

    async def plan(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("content_steps", 0) >= settings.content_budget:
            return _done(state.get("draft") or "", list(state.get("gaps") or ["budget"]))
        if not state.get("job_id"):
            return await _book_or_agent(state, model, parser, embedder, settings)
        question = str(state.get("question") or "")
        axes = list(state.get("axes") or [])
        reply = str(state.get("human_reply") or "").strip()
        text = reply or question
        # Empty text is the hand-off after «Другой файл». The old sentence must not replay.
        if not text.strip():
            return _goal_pause(state)
        if ".xlsx" in text.casefold() or ".xlsm" in text.casefold():
            loaded = await _succeeded_jobs(parser)
            if isinstance(loaded, dict):
                return loaded
            named_jobs = _named_jobs(text, loaded)
            if len(named_jobs) > 1:
                return _job_pause(loaded)
            if len(named_jobs) == 1:
                bound = await _bind_named_file(state, text, named_jobs[0], axes, parser)
                if bound is not None:
                    return bound
        found = await _label_rows_for(state, parser, label_rows)
        if isinstance(found, dict):
            return found
        filename = str((state.get("summary") or {}).get("source_filename") or "").strip()
        filenames = [filename] if filename else []
        pool = choice_rows(text, found, filenames)
        if not pool:
            return await _move(
                state, text, found, filenames, parser, model, embedder, settings
            )
        return await _rank_labels(
            state, text, pool, found, filenames, ranker, parser, settings
        )

    async def search(state: dict[str, Any]) -> dict[str, Any]:
        job_id = state["job_id"]
        try:
            etag = await parser.head_catalog_etag(job_id)
        except ParserError as exc:
            return _terminal_or_raise(exc)
        if state.get("book_etag") and etag and etag != state["book_etag"]:
            return {
                "reload": True,
                "citations": [],
                "selected": [],
                "search_again": False,
                "terminal": "",
            }
        plan_body = state.get("plan") or {}
        if (
            plan_body.get("source") == "question"
            and not list(plan_body.get("needles") or [])
            and state.get("selected")
        ):
            return {"awaiting": "", "terminal": "", "search_again": False}
        period_keys = {
            str(period.get("period_key")).casefold()
            for axis in state.get("axes") or []
            for period in axis.get("periods") or []
            if period.get("period_key")
        }
        try:
            pages = await _catalog_pages(
                parser,
                job_id,
                list(plan_body.get("needles") or []),
                period_keys,
                settings.catalog_search_limit,
            )
        except ParserError as exc:
            return _terminal_or_raise(exc)
        pages = [narrow_page(needle, page) for needle, page in pages]
        if plan_body.get("source") == "question" and _pages_empty(pages):
            return await _code_miss(state, model, settings, parser, ranker)
        if not pages:
            return _miss(state, "Пустой поиск.", settings)
        if (plan_body.get("question_type") or "lookup") == "compose":
            return _compose(state, pages, settings)
        return _single(state, pages, settings)

    async def bind(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("pending") == "goal":
            return await _bind_goal(state, parser, settings, label_rows)
        # A row pause with no menu is a period or a miss; that reply still goes to plan.
        if state.get("pending") == "row":
            offers = list(state.get("offers") or [])
            if not offers:
                try:
                    offers = await _offers_from_pause(state, parser, settings)
                except ParserError as exc:
                    return _terminal_or_raise(exc)
            if offers:
                return await _choose_offer({**state, "offers": offers}, settings)
        if state.get("pending") != "job":
            return {"pending": "", "terminal": "", "just_bound_row": False}
        offered = list(state.get("jobs") or [])
        reply = state.get("human_reply") or ""
        named = _named_jobs(reply, offered)
        if len(named) == 1:
            found = named[0]
        elif len(named) > 1:
            found = None
        else:
            found = _match_job(reply, offered)
        if found is None:
            if state.get("clarify_rounds", 0) >= settings.clarify_budget:
                return _done("Книга не выбрана.", ["job"])
            return _job_pause(list(state.get("jobs") or []), again=True)
        return {
            "job_id": found["job_id"],
            "pending": "",
            "awaiting": "",
            "human_reply": "",
            "user_question": "",
            "offers": [],
            "just_bound_job": True,
            "just_bound_row": False,
            "clarify_rounds": 0,
            "last_user_question": "",
            "terminal": "",
        }

    async def ask_user(state: dict[str, Any]) -> dict[str, Any]:
        # interrupt() is first: LangGraph replays the node from the start on resume.
        reply = interrupt(state.get("user_question") or "")
        return {
            "human_reply": reply,
            "awaiting": "",
            "clarify_rounds": state.get("clarify_rounds", 0) + 1,
            "terminal": "",
        }

    async def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        plan_body = state.get("plan") or {}
        selected = list(state.get("selected") or [])
        period_ids = list(state.get("period_ids") or [])
        depth = _depth(plan_body, settings)
        cap = settings.observation_cap
        per_call = max(1, cap // max(len(selected), 1))
        observations: list[dict[str, Any]] = []
        steps = list(state.get("steps") or [])
        try:
            for row in selected:
                document = await parser.get_observations(
                    state["job_id"],
                    row_key=row["row_key"],
                    period_ids=period_ids,
                    precedent_depth=depth,
                    limit=per_call,
                )
                batch = [
                    without_account_code(scalar_observation(item))
                    for item in (document.get("observations") or [])
                ]
                steps.append(
                    {
                        "kind": "observations",
                        "row_key": row["row_key"],
                        "period_ids": period_ids,
                        "precedent_depth": depth,
                        "ids": [f"{item.get('row_key')}:{item.get('period_id')}" for item in batch],
                    }
                )
                if document.get("truncated") and not period_ids:
                    return _ask(
                        state,
                        f"Ряд обрезан лимитом {cap}. Назовите период, "
                        "среднее по обрезанному ряду не считается.",
                        steps=steps,
                        settings=settings,
                    )
                observations.extend(batch)
            if wants_influence(state.get("question") or "", plan_body.get("trace") or ""):
                room = cap - len(observations)
                if room > 0:
                    observations.extend(
                        await _dependents(
                            parser, state, selected, period_ids, steps, room, settings
                        )
                    )
        except ParserError as exc:
            return _terminal_or_raise(exc)
        return {
            "observations": observations[:cap],
            "steps": steps,
            "awaiting": "",
            "terminal": "",
        }

    async def answer(state: dict[str, Any]) -> dict[str, Any]:
        if (state.get("plan") or {}).get("source") == "question":
            built = cache_answer(state)
            return {
                "draft": built["draft"],
                "proposed_citations": built["citations"],
                "draft_from_cache": True,
                "cache_missing": built["missing"],
                "schema_error": False,
                "terminal": "",
            }
        system, user = answer_messages(state)
        try:
            parsed = await model.complete_json(role="answer", system=system, user=user)
        except SchemaError:
            return {
                "schema_error": True,
                "gaps": ["schema"],
                "satisfactory": False,
                "draft": "Модель не вернула ответ.",
                "draft_from_cache": False,
                "cache_missing": False,
                "terminal": "",
            }
        return {
            "draft": str(parsed.get("text") or ""),
            "proposed_citations": list(parsed.get("citations") or []),
            "draft_from_cache": False,
            "cache_missing": False,
            "schema_error": False,
            "terminal": "",
        }

    async def check(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("schema_error"):
            return {
                "gaps": ["schema"],
                "gap_keys": ["schema"],
                "same_gap": True,
                "satisfactory": False,
                "schema_error": False,
                "terminal": "",
            }
        plan_body = state.get("plan") or {}
        if state.get("cache_missing"):
            gaps = ["number:missing"]
            return {
                "gaps": gaps,
                "gap_keys": gaps,
                "same_gap": True,
                "satisfactory": False,
                "citations": [],
                "terminal": "",
            }
        proposed = list(state.get("proposed_citations") or [])
        observed = list(state.get("observations") or [])
        if state.get("draft_from_cache"):
            return {
                "gaps": [],
                "gap_keys": [],
                "same_gap": False,
                "satisfactory": True,
                "citations": _accepted_citations(proposed, observed),
                "terminal": "",
            }
        code_gaps = verify_answer(
            question=state.get("question") or "",
            question_type=plan_body.get("question_type") or "lookup",
            trace=plan_body.get("trace") or "none",
            text=state.get("draft") or "",
            citations=proposed,
            observations=observed,
        )
        scale_only = bool(code_gaps) and all(str(item).startswith("scale:") for item in code_gaps)
        if scale_only or not code_gaps:
            return {
                "gaps": [],
                "gap_keys": [],
                "same_gap": False,
                "satisfactory": True,
                "citations": _accepted_citations(proposed, observed),
                "terminal": "",
            }
        return {
            "gaps": code_gaps,
            "gap_keys": code_gaps,
            "same_gap": code_gaps == (state.get("gap_keys") or []),
            "satisfactory": False,
            "terminal": "",
        }

    async def close(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("draft_from_cache") and not state.get("cache_missing"):
            steps = list(state.get("steps") or [])
            steps.append({"kind": "verdict", "satisfactory": True, "gaps": []})
            return {
                "draft": state.get("draft") or "",
                "gaps": [],
                "satisfactory": True,
                "steps": steps,
                "terminal": "done",
                "awaiting": "",
            }
        if state.get("cache_missing"):
            steps = list(state.get("steps") or [])
            steps.append({"kind": "verdict", "satisfactory": False, "gaps": []})
            return {
                "draft": state.get("draft") or "Подтверждённого числа в срезе нет.",
                "gaps": [],
                "satisfactory": False,
                "steps": steps,
                "terminal": "done",
                "awaiting": "",
            }
        gaps = [str(item) for item in (state.get("gaps") or [])]
        satisfactory = bool(state.get("satisfactory")) and not gaps
        draft = state.get("draft") or ""
        if not satisfactory:
            if any(item.startswith("number:") for item in gaps):
                draft = "Подтверждённого числа в срезе нет."
            visible = gaps
            if state.get("citations"):
                visible = [item for item in gaps if not item.startswith("scale:")]
            if visible:
                draft = f"{draft}\nНе хватает: {'; '.join(visible)}".strip()
            elif not draft:
                draft = "Ответ не собран."
        steps = list(state.get("steps") or [])
        steps.append({"kind": "verdict", "satisfactory": satisfactory, "gaps": gaps})
        return {
            "draft": draft,
            "satisfactory": satisfactory,
            "steps": steps,
            "terminal": "done",
            "awaiting": "",
        }

    builder = StateGraph(AgentState)
    builder.add_node("load", load)
    builder.add_node("plan", plan)
    builder.add_node("search", search)
    builder.add_node("bind", bind)
    builder.add_node("ask_user", ask_user)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_node("check", check)
    builder.add_node("close", close)
    builder.add_edge(START, "load")
    builder.add_conditional_edges(
        "load",
        _route_load,
        {"ask_user": "ask_user", "plan": "plan", "close": "close"},
    )
    builder.add_conditional_edges(
        "plan",
        _route_plan,
        {"search": "search", "ask_user": "ask_user", "close": "close", "load": "load"},
    )
    builder.add_conditional_edges(
        "search",
        _route_search,
        {
            "load": "load",
            "plan": "plan",
            "ask_user": "ask_user",
            "retrieve": "retrieve",
            "close": "close",
        },
    )
    builder.add_edge("ask_user", "bind")
    builder.add_conditional_edges(
        "bind",
        _route_bind,
        {
            "ask_user": "ask_user",
            "load": "load",
            "plan": "plan",
            "retrieve": "retrieve",
            "close": "close",
        },
    )
    builder.add_conditional_edges(
        "retrieve",
        _route_retrieve,
        {"ask_user": "ask_user", "answer": "answer", "close": "close"},
    )
    builder.add_edge("answer", "check")

    def route_check(state: dict[str, Any]) -> str:
        if state.get("draft_from_cache") and not state.get("cache_missing"):
            return "close"
        if state.get("cache_missing"):
            return "close"
        gaps = [str(item) for item in (state.get("gaps") or [])]
        if not gaps or gaps == ["schema"]:
            return "close"
        if state.get("same_gap") or state.get("content_steps", 0) >= settings.content_budget:
            return "close"
        return "plan"

    builder.add_conditional_edges("check", route_check, {"plan": "plan", "close": "close"})
    builder.add_edge("close", END)
    return builder.compile(checkpointer=checkpointer)


def _route_load(state: dict[str, Any]) -> str:
    if state.get("terminal"):
        return "close"
    if state.get("awaiting") == "job":
        return "ask_user"
    return "plan"


def _route_plan(state: dict[str, Any]) -> str:
    if state.get("terminal"):
        return "close"
    if state.get("awaiting"):
        return "ask_user"
    if state.get("just_bound_job"):
        return "load"
    return "search"


def _route_search(state: dict[str, Any]) -> str:
    if state.get("terminal"):
        return "close"
    if state.get("reload"):
        return "load"
    if state.get("search_again"):
        return "plan"
    if state.get("awaiting"):
        return "ask_user"
    return "retrieve"


def _route_bind(state: dict[str, Any]) -> str:
    if state.get("terminal"):
        return "close"
    if state.get("awaiting"):
        return "ask_user"
    if state.get("just_bound_job"):
        return "load"
    if state.get("just_bound_row"):
        return "retrieve"
    return "plan"


def _route_retrieve(state: dict[str, Any]) -> str:
    if state.get("terminal"):
        return "close"
    if state.get("awaiting"):
        return "ask_user"
    return "answer"


def _terminal_or_raise(exc: ParserError) -> dict[str, Any]:
    if exc.status == 409 and exc.code == "report_not_ready":
        return _done("Книга ещё собирается.", ["report_not_ready"])
    if exc.status == 404:
        return _done("Книга не найдена.", ["not_found"])
    raise exc


def _done(draft: str, gaps: list[str]) -> dict[str, Any]:
    return {
        "draft": draft,
        "gaps": gaps,
        "satisfactory": False,
        "terminal": "partial",
        "awaiting": "",
        "search_again": False,
        "reload": False,
    }


def _depth(plan_body: dict[str, Any], settings: Settings) -> int:
    if plan_body.get("question_type") == "explain" or plan_body.get("trace") == "precedents":
        return settings.precedent_depth
    return 0


def _pages_empty(pages: list[tuple[str, dict[str, Any]]]) -> bool:
    return all(int(page.get("total") or 0) == 0 for _needle, page in pages)


def _agent_reply() -> dict[str, Any]:
    """A turn about the agent. No passport, no job list, and no cell figures."""
    return {
        "draft": chitchat_reply(),
        "gaps": [],
        "satisfactory": True,
        "citations": [],
        "terminal": "done",
        "awaiting": "",
        "search_again": False,
        "reload": False,
        "draft_from_cache": False,
        "cache_missing": False,
        "user_question": "",
        "pending": "",
        "offers": [],
        "selected": [],
        "period_ids": [],
    }


def _act_names(parsed: Any) -> list[str]:
    acts = parsed.get("acts") if isinstance(parsed, dict) else None
    if not isinstance(acts, list):
        return []
    return [str(item) for item in acts]


def _goal_pause(state: dict[str, Any]) -> dict[str, Any]:
    """Ask what to look at. The open book stays, and the draft is not the answer."""
    filename = str((state.get("summary") or {}).get("source_filename") or "")
    question = goal_question(filename)
    return {
        "user_question": question,
        "menu_question": question,
        "offers": [{"act": act, "label": label} for act, label in GOAL_CHOICES],
        "awaiting": "goal",
        "pending": "goal",
        "clarify_rounds": 0,
        "terminal": "",
        "just_bound_job": False,
        "just_bound_row": False,
        "human_reply": "",
    }


def _goal_closed() -> dict[str, Any]:
    """Drop the goal menu and keep the reply for plan."""
    return {
        "offers": [],
        "pending": "",
        "awaiting": "",
        "user_question": "",
        "menu_question": "",
        "just_bound_row": False,
        "just_bound_job": False,
        "terminal": "",
        "clarify_rounds": 0,
    }


async def _bind_goal(
    state: dict[str, Any],
    parser: ParserClient,
    settings: Settings,
    label_rows: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """A number or a printed line runs that action. Anything else is a new turn."""
    reply = str(state.get("human_reply") or "")
    folded = reply.strip().casefold()
    shown = {
        str(state.get("menu_question") or "").strip().casefold(),
        str(state.get("user_question") or "").strip().casefold(),
    }
    shown.discard("")
    if not folded or folded in shown:
        return _goal_pause(state)
    choice = match_goal(reply)
    job_id = str(state.get("job_id") or "")
    if choice == "overview":
        opened = await _overview_of(parser, job_id)
        if not opened.get("satisfactory"):
            return opened
        return opened | {
            "last_act": "overview",
            "offers": [],
            "pending": "",
            "awaiting": "",
            "menu_question": "",
            "user_question": "",
            "human_reply": "",
            "clarify_rounds": 0,
        }
    if choice == "catalog":
        found = await _label_rows_for(state, parser, label_rows)
        if isinstance(found, dict):
            return found
        draft = catalog_reply(found, "")
        if not draft:
            return _missing(state, settings, "catalog")
        return _finished(draft, "catalog", job_id) | {"clarify_rounds": 0}
    if choice == "figure":
        return _ask(state, "Назовите подпись.", settings=settings) | {"clarify_rounds": 0}
    if choice == "files":
        loaded = await _succeeded_jobs(parser)
        if isinstance(loaded, dict):
            return loaded
        return _job_pause(loaded) | {
            "question": "",
            "last_act": "",
            "menu_question": "",
            "offers": [],
            "clarify_rounds": 0,
        }
    return _goal_closed()


def _job_pause(jobs: list[dict[str, Any]], *, again: bool = False) -> dict[str, Any]:
    if again:
        question = "Не нашёл такую книгу. Назовите файл ещё раз."
    else:
        lines = [f"- {job.get('source_filename') or job.get('job_id')}" for job in jobs]
        question = "Какую книгу открыть?\n" + "\n".join(lines)
    return {
        "jobs": jobs,
        "pending": "job",
        "awaiting": "job",
        "user_question": question,
        "reload": False,
        "just_bound_job": False,
        "just_bound_row": False,
        "terminal": "",
    }


async def _succeeded_jobs(parser: ParserClient) -> list[dict[str, Any]] | dict[str, Any]:
    try:
        return await parser.list_jobs(status="succeeded")
    except ParserError as exc:
        return _terminal_or_raise(exc)


async def _job_menu(parser: ParserClient) -> dict[str, Any]:
    jobs = await _succeeded_jobs(parser)
    if isinstance(jobs, dict):
        return jobs
    if not jobs:
        return _done("Готовых книг нет.", ["no_jobs"])
    return _job_pause(jobs)


def _books_done(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """The finished file list. The next sentence is classified again."""
    if not jobs:
        return _done("Готовых книг нет.", ["no_jobs"])
    return {
        "draft": books_reply(jobs),
        "gaps": [],
        "satisfactory": True,
        "citations": [],
        "terminal": "done",
        "awaiting": "",
        "search_again": False,
        "reload": False,
        "draft_from_cache": False,
        "cache_missing": False,
        "user_question": "",
        "pending": "",
        "offers": [],
        "selected": [],
        "period_ids": [],
    }


async def _books_answer(
    parser: ParserClient, jobs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if jobs is None:
        loaded = await _succeeded_jobs(parser)
        if isinstance(loaded, dict):
            return loaded
        jobs = loaded
    return _books_done(jobs)


def _filenames(jobs: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for job in jobs:
        name = str(job.get("source_filename") or job.get("job_id") or "").strip()
        if name:
            names.append(name)
    return names


def _is_search_plan(found: dict[str, Any]) -> bool:
    return bool(found.get("plan")) and not found.get("terminal") and not found.get("awaiting")


async def _overview_of(parser: ParserClient, job_id: str) -> dict[str, Any]:
    """The stored overview of one book. The previous row does not carry over."""
    try:
        document = await parser.get_context(job_id)
    except ParserError as exc:
        return _terminal_or_raise(exc)
    return {
        "draft": book_overview(document),
        "gaps": [],
        "satisfactory": True,
        "citations": [],
        "terminal": "done",
        "awaiting": "",
        "search_again": False,
        "reload": False,
        "job_id": job_id,
        "selected": [],
        "period_ids": [],
        "human_reply": "",
        "user_question": "",
        "pending": "",
        "label_reply": False,
        "draft_from_cache": False,
        "cache_missing": False,
    }


async def _about_acts(model: JsonModel, text: str, filenames: list[str]) -> list[str]:
    system, user = about_messages(text, filenames)
    try:
        parsed = await model.complete_json(role="about", system=system, user=user)
    except SchemaError:
        return []
    return _act_names(parsed)


async def _named_file_turn(
    text: str,
    state: dict[str, Any],
    model: JsonModel,
    parser: ParserClient,
    job: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """One real file is named. The choice runs after the book loads."""
    del state, model, parser, jobs
    job_id = str(job.get("job_id") or "")
    return {
        "job_id": job_id,
        "just_bound_job": True,
        "question": text,
        "human_reply": "",
        "pending": "",
        "awaiting": "",
        "terminal": "",
        "selected": [],
        "period_ids": [],
        "offers": [],
        "user_question": "",
        "draft_from_cache": False,
        "cache_missing": False,
    }


async def _dialog_turn(
    text: str,
    state: dict[str, Any],
    model: JsonModel,
    parser: ParserClient,
    *,
    jobs: list[dict[str, Any]] | None = None,
    unbound_row: str = "search",
) -> dict[str, Any] | None:
    """Ground a workbook, then an act. None means the label search continues."""
    if jobs is None:
        loaded = await _succeeded_jobs(parser)
        if isinstance(loaded, dict):
            return loaded
        jobs = loaded
    named = _named_jobs(text, jobs)
    if len(named) > 1:
        return _job_pause(jobs)
    if len(named) == 1:
        return await _named_file_turn(text, state, model, parser, named[0], jobs)
    acts = await _about_acts(model, text, _filenames(jobs))
    if "books" in acts:
        return _books_done(jobs)
    if "book" in acts:
        current = str(state.get("job_id") or "")
        if current:
            return await _overview_of(parser, current)
        return _job_pause(jobs)
    if "row" in acts:
        if unbound_row == "menu":
            return _job_pause(jobs)
        return None
    return _agent_reply()


async def _book_or_agent(
    state: dict[str, Any],
    model: JsonModel,
    parser: ParserClient,
    embedder: Any,
    settings: Settings,
) -> dict[str, Any]:
    """No book is bound. One filename loads that book. Otherwise the act classifier."""
    question = str(state.get("question") or "")
    jobs = await _succeeded_jobs(parser)
    if isinstance(jobs, dict):
        return jobs
    named = _named_jobs(question, jobs)
    if len(named) > 1:
        return _job_pause(jobs)
    if len(named) == 1:
        return await _named_file_turn(question, state, model, parser, named[0], jobs)
    return await _move(state, question, [], [], parser, model, embedder, settings, jobs)


async def _dialog_choice(
    text: str, jobs: list[dict[str, Any]], ranker: Any
) -> dict[str, Any]:
    """books, book, intro, or none. No catalog row is in this choice."""
    if ranker is None:
        raise ModelError("ranker")
    criteria = {
        "books": _ASK_BOOKS,
        "book": _ASK_BOOK,
        "intro": _ASK_INTRO,
        "none": _ASK_NONE,
    }
    choice = await ranker.choose(text, criteria)
    decision = decide_margin(choice.probabilities, set(), menu_open=False)
    if decision.get("act") == "take":
        key = str(decision.get("key") or "")
        if key == "books":
            return _books_done(jobs)
        if key == "intro":
            return _agent_reply()
        if key not in {"book", "none"}:
            raise ModelError("ranker choice")
    return _job_pause(jobs)


async def _code_miss(
    state: dict[str, Any], model: JsonModel, settings: Settings, parser: Any, ranker: Any
) -> dict[str, Any]:
    """A code plan grounded nothing. A weak score does not open a book."""
    del model
    text = str(state.get("human_reply") or "").strip() or str(state.get("question") or "")
    if ranker is None:
        raise ModelError("ranker")
    criteria = {
        "books": _ASK_BOOKS,
        "book": _ASK_BOOK,
        "intro": _ASK_INTRO,
        "none": _ASK_NONE,
    }
    choice = await ranker.choose(text, criteria)
    decision = decide_margin(choice.probabilities, set(), menu_open=False)
    if decision.get("act") == "take":
        key = str(decision.get("key") or "")
        if key == "books":
            return await _books_answer(parser)
        if key == "intro":
            return _agent_reply()
        if key not in {"book", "none"}:
            raise ModelError("ranker choice")
    return _ask(state, _MISSING_ROW, settings=settings) | {
        "label_reply": True,
        "draft_from_cache": False,
        "cache_missing": False,
    }


def _miss(state: dict[str, Any], note: str, settings: Settings) -> dict[str, Any]:
    misses = state.get("search_misses", 0) + 1
    if misses >= 2:
        return _ask(
            state,
            _MISSING_ROW,
            settings=settings,
        ) | {
            "search_misses": misses,
            "search_note": note,
        }
    return {
        "search_misses": misses,
        "search_again": True,
        "search_note": note,
        "awaiting": "",
        "terminal": "",
    }


def _single(
    state: dict[str, Any], pages: list[tuple[str, dict[str, Any]]], settings: Settings
) -> dict[str, Any]:
    if all(int(page.get("total") or 0) == 0 for _needle, page in pages):
        note = ", ".join(needle for needle, _page in pages)
        return _miss(state, f"Нет совпадений: {note}.", settings)
    useful = [(needle, page) for needle, page in pages if int(page.get("total") or 0) > 0]
    rows = _distinct_rows(useful)
    if any(int(page.get("total") or 0) != 1 for _needle, page in useful) or len(rows) != 1:
        return _menu(useful)
    return _select(state, rows, settings)


def _compose(
    state: dict[str, Any], pages: list[tuple[str, dict[str, Any]]], settings: Settings
) -> dict[str, Any]:
    if all(int(page.get("total") or 0) == 0 for _needle, page in pages):
        return _miss(state, "Нет совпадений по метрикам.", settings)
    pending = [(needle, page) for needle, page in pages if int(page.get("total") or 0) != 1]
    if pending:
        return _menu(pending)
    return _select(state, _distinct_rows(pages), settings)


def _menu(pages: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    question, rows = render_menu(pages)
    return _show_menu(question, [compact_row(row) for row in rows])


def _shown_needles(question: str) -> list[str]:
    text = question.strip()
    if not text.startswith(_MENU_LEAD):
        return []
    return [item.strip() for item in _MENU_NEEDLE.findall(text) if item.strip()]


async def _offers_from_pause(
    state: dict[str, Any], parser: ParserClient, settings: Settings
) -> list[dict[str, Any]]:
    """Rows of a menu saved before offers existed. The reply itself is not the query."""
    needles = _shown_needles(str(state.get("user_question") or ""))
    job_id = str(state.get("job_id") or "")
    if not needles or not job_id:
        return []
    period_keys = {
        str(period.get("period_key")).casefold()
        for axis in state.get("axes") or []
        for period in axis.get("periods") or []
        if period.get("period_key")
    }
    pages = await _catalog_pages(
        parser,
        job_id,
        needles,
        period_keys,
        settings.catalog_search_limit,
    )
    pages = [narrow_page(needle, page) for needle, page in pages]
    return [compact_row(row) for row in _distinct_rows(pages)]


def _same_question(state: dict[str, Any]) -> bool:
    reply = str(state.get("human_reply") or "").strip().casefold()
    question = str(state.get("question") or "").strip().casefold()
    return bool(reply) and reply == question


def _show_menu(question: str, offers: list[dict[str, Any]]) -> dict[str, Any]:
    """A row menu starts its own reply count. The budget check in _ask does not run."""
    return {
        "user_question": question,
        "last_user_question": question,
        "menu_question": question,
        "offers": offers,
        "clarify_rounds": 0,
        "awaiting": "row",
        "pending": "row",
        "just_bound_row": False,
        "just_bound_job": False,
        "human_reply": "",
        "terminal": "",
        "search_again": False,
    }


def _menu_again(state: dict[str, Any], offers: list[dict[str, Any]]) -> dict[str, Any]:
    """The user repeated the question. Show the menu without spending the clarify budget."""
    stored = str(state.get("menu_question") or "").strip()
    current = str(state.get("user_question") or "").strip()
    if stored:
        text = stored
    elif current.startswith(_MENU_LEAD):
        text = current
    else:
        text = f"{MENU_INSTRUCTION}\n{offered_labels(offers)}"
    return _show_menu(text, offers)


def _label_menu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    offers = [compact_row(row) for row in rows]
    labels = offered_labels(offers)
    question = f"{MENU_INSTRUCTION}\n{labels}" if labels else MENU_INSTRUCTION
    return _show_menu(question, offers)


def _select_ready(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    named: list[dict[str, str]],
    settings: Settings,
) -> dict[str, Any]:
    labels = [str(row.get("label") or "") for row in rows]
    planned = _question_plan(state, labels, named)
    planned["plan"]["needles"] = []
    planned["draft_from_cache"] = False
    planned["cache_missing"] = False
    planned["label_reply"] = False
    chosen = _select({**state, **planned}, rows, settings)
    if chosen.get("awaiting") or chosen.get("terminal"):
        return chosen | {"draft_from_cache": False, "cache_missing": False}
    return {**planned, **chosen}


def _from_hits(
    state: dict[str, Any],
    hits: list[dict[str, Any]],
    named: list[dict[str, str]],
    settings: Settings,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in hits:
        label = str(row.get("label") or "").strip().casefold()
        if label not in groups:
            order.append(label)
            groups[label] = []
        groups[label].append(row)
    ambiguous = [row for label in order if len(groups[label]) > 1 for row in groups[label]]
    if ambiguous:
        return _label_menu(ambiguous)
    blocked = _label_cap(state, len(hits), settings)
    if blocked:
        return blocked
    return _select_ready(state, hits, named, settings)


async def _label_rows_for(
    state: dict[str, Any],
    parser: ParserClient,
    cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]] | dict[str, Any]:
    job_id = str(state.get("job_id") or "")
    etag = str(state.get("book_etag") or "")
    key = (job_id, etag)
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        rows = await parser.list_rows(job_id)
    except ParserError as exc:
        return _terminal_or_raise(exc)
    for old in [item for item in cache if item[0] == job_id and item != key]:
        del cache[old]
    cache[key] = rows
    return rows


async def _bind_named_file(
    state: dict[str, Any],
    text: str,
    job: dict[str, Any],
    axes: list[dict[str, Any]],
    parser: ParserClient,
) -> dict[str, Any] | None:
    """Switch to another book. The open book stays, and the choice runs next."""
    del axes, parser
    job_id = str(job.get("job_id") or "")
    if job_id and job_id != str(state.get("job_id") or ""):
        return {
            "job_id": job_id,
            "just_bound_job": True,
            "pending": "",
            "awaiting": "",
            "human_reply": "",
            "user_question": "",
            "offers": [],
            "terminal": "",
            "selected": [],
            "period_ids": [],
            "question": text,
            "draft_from_cache": False,
            "cache_missing": False,
        }
    return None


def _choice_criteria(
    pool: list[dict[str, Any]], state: dict[str, Any]
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """One criterion per pooled row, then the dialog actions. Sheet only on a tie."""
    shared: dict[str, int] = {}
    for row in pool:
        label = str(row.get("label") or "").strip().casefold()
        shared[label] = shared.get(label, 0) + 1
    criteria: dict[str, str] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(pool):
        key = f"r{index}"
        label = str(row.get("label") or "").strip()
        text = label
        if shared.get(label.casefold(), 0) > 1:
            sheet = str(row.get("sheet") or "").strip()
            text = f"{label} Лист {sheet}."
        criteria[key] = text
        by_key[key] = row
    if pool:
        return criteria, by_key
    criteria["books"] = _ASK_BOOKS
    criteria["book"] = _ASK_BOOK
    criteria["intro"] = _ASK_INTRO
    criteria["none"] = _ASK_NONE
    held = [row for row in (state.get("selected") or []) if isinstance(row, dict)]
    if held:
        criteria["open"] = f"Открытая строка: {str(held[0].get('label') or '').strip()}"
    return criteria, by_key


def _named_periods(state: dict[str, Any], text: str) -> list[dict[str, str]]:
    axes = list(state.get("axes") or [])
    named = periods_from_question(text, axes)
    if str(state.get("human_reply") or "").strip() and not named:
        named = periods_from_question(str(state.get("question") or ""), axes)
    return named


def _sheet_list(
    state: dict[str, Any],
    text: str,
    catalog_rows: list[dict[str, Any]],
    filenames: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """The matching labels and their sheets. The turn ends without a number."""
    shown, total = inventory_rows(text, catalog_rows, filenames)
    if not shown:
        return _ask(state, _MISSING_ROW, settings=settings) | {
            "label_reply": True,
            "draft_from_cache": False,
            "cache_missing": False,
        }
    lines: list[str] = []
    for row in shown:
        label = str(row.get("label") or "").strip()
        lines.append(label)
        sheet = str(row.get("sheet") or "").strip()
        heading = _heading(row.get("label_path"), label)
        detail: list[str] = []
        if sheet:
            detail.append(f"Лист {sheet}.")
        if heading:
            detail.append(f"Раздел {heading}.")
        if detail:
            lines.append(" ".join(detail))
    if total > len(shown):
        lines.append(f"Показаны первые {len(shown)} из {total}.")
    return {
        "draft": "\n".join(lines),
        "gaps": [],
        "satisfactory": True,
        "citations": [],
        "terminal": "done",
        "awaiting": "",
        "offers": [],
        "pending": "",
        "user_question": "",
        "menu_question": "",
        "selected": [],
        "period_ids": [],
        "human_reply": "",
        "label_reply": False,
        "draft_from_cache": False,
        "cache_missing": False,
        "search_again": False,
        "just_bound_row": False,
        "just_bound_job": False,
        "last_act": "catalog",
    }


async def _rank_labels(
    state: dict[str, Any],
    text: str,
    pool: list[dict[str, Any]],
    catalog_rows: list[dict[str, Any]],
    filenames: list[str],
    ranker: Any,
    parser: ParserClient,
    settings: Settings,
) -> dict[str, Any]:
    """One choice. The returned key is the row, the overview, or the pause."""
    if ranker is None:
        raise ModelError("ranker")
    criteria, by_key = _choice_criteria(pool, state)
    filename = str((state.get("summary") or {}).get("source_filename") or "")
    choice = await ranker.choose(
        ranker_state(filename, text, bool(pool)),
        criteria,
        ask_act=bool(pool),
        ask_hold=not pool and bool(filename.strip()),
    )
    if pool and choice.sheets is not None and choice.sheets >= LIST_FLOOR:
        return _sheet_list(state, text, catalog_rows, filenames, settings)
    decision = decide_margin(choice.probabilities, set(by_key), menu_open=False)
    if decision.get("act") == "take":
        key = str(decision.get("key") or "")
        named = _named_periods(state, text)
        if key in by_key:
            return _select_ready(state, [by_key[key]], named, settings) | {"last_act": "figure"}
        if key == "books":
            return await _books_answer(parser)
        if key == "book":
            return await _overview_of(parser, str(state.get("job_id") or ""))
        if key == "intro":
            return _agent_reply()
        if key == "open":
            held = [row for row in (state.get("selected") or []) if isinstance(row, dict)]
            return _continue_selected(state, held, named, text, settings)
        if key != "none":
            raise ModelError("ranker choice")
    if decision.get("act") == "menu":
        rows = [by_key[key] for key in decision.get("keys") or [] if key in by_key]
        if rows:
            return _label_menu(rows)
    if decision.get("act") == "miss" and choice.hold is not None and choice.hold >= HOLD_FLOOR:
        return _agent_reply()
    return _ask(state, _MISSING_ROW, settings=settings) | {
        "label_reply": True,
        "draft_from_cache": False,
        "cache_missing": False,
    }


def _unique_leader(scores: dict[str, float] | None) -> str | None:
    """The single highest act, even under the floor. A tie has no leader."""
    if not scores:
        return None
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    act, top = ordered[0]
    if act not in ACTS:
        return None
    if len(ordered) > 1 and abs(top - ordered[1][1]) < 1e-9:
        return None
    return str(act)


def _finished(draft: str, act: str, job_id: str) -> dict[str, Any]:
    """A printed act. The menu is closed and the next turn is classified again."""
    body: dict[str, Any] = {
        "draft": draft,
        "gaps": [],
        "satisfactory": True,
        "citations": [],
        "terminal": "done",
        "awaiting": "",
        "offers": [],
        "pending": "",
        "user_question": "",
        "menu_question": "",
        "selected": [],
        "period_ids": [],
        "human_reply": "",
        "label_reply": False,
        "draft_from_cache": False,
        "cache_missing": False,
        "search_again": False,
        "just_bound_row": False,
        "just_bound_job": False,
        "last_act": act,
    }
    if job_id:
        body["job_id"] = job_id
    return body


def _missing(state: dict[str, Any], settings: Settings, act: str) -> dict[str, Any]:
    return _ask(state, _MISSING_ROW, settings=settings) | {
        "label_reply": True,
        "draft_from_cache": False,
        "cache_missing": False,
        "last_act": act,
    }


def _clarify(
    state: dict[str, Any], settings: Settings, text: str = _CLARIFY
) -> dict[str, Any]:
    return _ask(state, text, settings=settings) | {
        "label_reply": True,
        "draft_from_cache": False,
        "cache_missing": False,
        "last_act": "unclear",
    }


async def _need_book(
    parser: ParserClient, jobs: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Ask which file to open. The carried question is classified again inside it."""
    if jobs is None:
        loaded = await _succeeded_jobs(parser)
        if isinstance(loaded, dict):
            return loaded
        jobs = loaded
    if not jobs:
        return _done("Готовых книг нет.", ["no_jobs"])
    return _job_pause(jobs) | {"last_act": ""}


async def _emit(
    act: str,
    state: dict[str, Any],
    text: str,
    rows: list[dict[str, Any]],
    parser: ParserClient,
    settings: Settings,
    jobs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Print the act the classifier or the chat already chose. No stem can replace it."""
    job_id = str(state.get("job_id") or "")
    held = [row for row in (state.get("selected") or []) if isinstance(row, dict)]
    if act == "files":
        loaded = jobs
        if loaded is None:
            loaded = await _succeeded_jobs(parser)
            if isinstance(loaded, dict):
                return loaded
        named = _named_jobs(text, loaded)
        same_book = (
            bool(job_id)
            and len(named) == 1
            and str(named[0].get("job_id") or "") == job_id
        )
        if same_book:
            return _goal_pause(state)
        return await _books_answer(parser, loaded) | {"last_act": "files"}
    if act == "overview":
        if job_id:
            return await _overview_of(parser, job_id) | {"last_act": "overview"}
        return await _need_book(parser, jobs)
    if act == "catalog":
        if not job_id:
            return await _need_book(parser, jobs)
        draft = catalog_reply(rows, text)
        if not draft:
            return _missing(state, settings, "catalog")
        return _finished(draft, "catalog", job_id)
    if act == "figure":
        if job_id:
            return _missing(state, settings, "figure")
        return await _need_book(parser, jobs)
    if act == "explain":
        if held and job_id:
            continued = _continue_selected(state, held, [], text, settings)
            if "plan" in continued:
                body = dict(continued["plan"])
                body["question_type"] = "explain"
                body["trace"] = "precedents"
                return continued | {"plan": body, "last_act": "explain"}
            return continued
        if job_id:
            return _ask(state, _EXPLAIN_ASK, settings=settings) | {
                "label_reply": True,
                "draft_from_cache": False,
                "cache_missing": False,
                "last_act": "explain",
            }
        return await _need_book(parser, jobs)
    if act == "greet":
        if job_id:
            return _goal_pause(state) | {"last_act": "greet"}
        if jobs is None:
            loaded = await _succeeded_jobs(parser)
            if isinstance(loaded, dict):
                return loaded
            jobs = loaded
        if jobs:
            return _job_pause(jobs) | {"last_act": ""}
        return _agent_reply() | {"last_act": "greet"}
    return _clarify(state, settings)


async def _move(
    state: dict[str, Any],
    text: str,
    catalog_rows: list[dict[str, Any]],
    filenames: list[str],
    parser: ParserClient,
    model: JsonModel,
    embedder: Any,
    settings: Settings,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """No pooled label. A period fills the open row; otherwise the embedding chooses the act."""
    held = [row for row in (state.get("selected") or []) if isinstance(row, dict)]
    period = sole_period_key(text, filenames, list(state.get("axes") or []), held)
    if period:
        continued = _continue_selected(
            state, held, [{"period_key": period}], text, settings
        )
        if "plan" in continued:
            return continued | {"last_act": "figure"}
        return continued
    book_open = bool(state.get("job_id"))
    rendered = act_text(
        "открыта" if book_open else "закрыта",
        str(state.get("last_act") or "") or "пусто",
        "открыта" if held else "нет",
        text,
    )
    scores: dict[str, float] | None = None
    if embedder is not None:
        try:
            scores = await embedder.score(rendered)
        except ModelError:
            scores = None
    decided = prototype_decision(scores)
    if decided is not None:
        act, top, gap = decided
        logger.info("embed %s %.3f %.3f", act, top, gap)
        return await _emit(act, state, text, catalog_rows, parser, settings, jobs)
    if scores:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top = ordered[0][1]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        logger.info("embed abstain %.3f %.3f", top, top - second)
    else:
        logger.info("embed abstain")
    system, user = talk_messages(
        "открыта" if book_open else "закрыта",
        str(state.get("last_act") or ""),
        bool(held),
        text,
    )
    try:
        parsed = await model.complete_json(role="talk", system=system, user=user)
    except SchemaError:
        return _clarify(state, settings)
    act = str(parsed.get("act") or "")
    leader = _unique_leader(scores)
    if act not in ACTS or (leader is not None and act != leader):
        return _clarify(state, settings)
    logger.info("talk %s", act)
    return await _emit(act, state, text, catalog_rows, parser, settings, jobs)


async def _choose_offer(state: dict[str, Any], settings: Settings) -> dict[str, Any]:
    offers = [row for row in (state.get("offers") or []) if isinstance(row, dict)]
    if _same_question(state):
        return _menu_again(state, offers)
    matched = matching_rows(str(state.get("human_reply") or ""), offers)
    if len(matched) == 1:
        # The menu question is answered. The next pause, including a period, starts at zero.
        fresh = {**state, "clarify_rounds": 0}
        named = periods_from_question(
            str(state.get("human_reply") or ""), list(state.get("axes") or [])
        )
        if not named:
            named = periods_from_question(
                str(state.get("question") or ""), list(state.get("axes") or [])
            )
        chosen = _select_ready(fresh, matched, named, settings)
        if chosen.get("awaiting") or chosen.get("terminal"):
            return chosen | {
                "just_bound_row": False,
                "just_bound_job": False,
                "clarify_rounds": 0,
            }
        return chosen | {
            "human_reply": "",
            "offers": [],
            "pending": "",
            "awaiting": "",
            "just_bound_row": True,
            "just_bound_job": False,
            "terminal": "",
            "clarify_rounds": 0,
        }
    if len(matched) > 1:
        question = f"{MENU_INSTRUCTION}\n{offered_labels(matched)}"
        return _show_menu(question, [compact_row(row) for row in matched])
    return {
        "offers": [],
        "pending": "",
        "awaiting": "",
        "user_question": "",
        "menu_question": "",
        "just_bound_row": False,
        "just_bound_job": False,
        "terminal": "",
    }


def _periods_for_rows(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    requested: list[Any],
    settings: Settings,
) -> dict[str, Any] | list[str]:
    """One shared period-key list, or the pause when the axis cannot decide."""
    period_ids: list[str] | None = None
    for row in rows:
        keys, error = resolve_periods(
            requested,
            list(state.get("axes") or []),
            list(row.get("axis_ids") or []),
        )
        if error == "many":
            return _ask(
                state, "Период подходит нескольким ключам оси. Назовите один.", settings=settings
            )
        if error == "none":
            return _ask(
                state,
                "Период не находится на оси строки. Назовите ключ или год.",
                settings=settings,
            )
        if period_ids is None:
            period_ids = keys
        elif keys != period_ids:
            return _ask(
                state,
                "Период не один на всех выбранных строках. Назовите ключ.",
                settings=settings,
            )
    period_ids = period_ids or []
    if not period_ids and periods_from_question(
        str(state.get("question") or ""), list(state.get("axes") or [])
    ):
        return _ask(
            state,
            "Период не находится на оси строки. Назовите ключ или год.",
            settings=settings,
        )
    return period_ids


def _continue_selected(
    state: dict[str, Any],
    held: list[dict[str, Any]],
    named: list[dict[str, str]],
    text: str,
    settings: Settings,
) -> dict[str, Any]:
    """A follow-up with no new label keeps the rows and skips the catalog."""
    why = asks_how(text)
    if named:
        resolved = _periods_for_rows(state, held, named, settings)
        if isinstance(resolved, dict):
            return resolved
        period_ids = resolved
    else:
        period_ids = [str(item) for item in (state.get("period_ids") or [])]
    return {
        "content_steps": state.get("content_steps", 0) + 1,
        "plan": {
            "question_type": "explain" if why else "lookup",
            "needles": [],
            "periods": named,
            "trace": "precedents" if why else "none",
            "source": "question",
        },
        "selected": held,
        "period_ids": period_ids,
        "human_reply": "",
        "search_again": False,
        "draft_from_cache": False,
        "cache_missing": False,
        "label_reply": False,
        "awaiting": "",
        "terminal": "",
    }


def _select(
    state: dict[str, Any], rows: list[dict[str, Any]], settings: Settings
) -> dict[str, Any]:
    plan_body = state.get("plan") or {}
    resolved = _periods_for_rows(state, rows, list(plan_body.get("periods") or []), settings)
    if isinstance(resolved, dict):
        return resolved
    period_ids = resolved
    selected = [compact_row(row) for row in rows]
    steps = list(state.get("steps") or [])
    steps.append(
        {
            "kind": "search",
            "row_keys": [row["row_key"] for row in selected],
            "period_ids": period_ids,
        }
    )
    return {
        "selected": selected,
        "period_ids": period_ids,
        "steps": steps,
        "awaiting": "",
        "pending": "",
        "search_again": False,
        "search_note": "",
        "search_misses": 0,
        "terminal": "",
    }


def _ask(
    state: dict[str, Any],
    question: str,
    steps: list[dict[str, Any]] | None = None,
    *,
    settings: Settings,
) -> dict[str, Any]:
    if (
        question == state.get("last_user_question")
        or state.get("clarify_rounds", 0) >= settings.clarify_budget
    ):
        return _done(state.get("draft") or "Не смог выбрать строку.", ["clarify"])
    patch: dict[str, Any] = {
        "user_question": question,
        "last_user_question": question,
        "awaiting": "row",
        "pending": "row",
        "offers": [],
        "menu_question": "",
        "just_bound_row": False,
        "search_again": False,
        "terminal": "",
    }
    if steps is not None:
        patch["steps"] = steps
    return patch


def _distinct_rows(pages: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _needle, page in pages:
        for row in page.get("rows") or []:
            key = str(row.get("row_key"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _label_cap(
    state: dict[str, Any], count: int, settings: Settings
) -> dict[str, Any] | None:
    """One answer holds observation_cap observations, so it cannot open more labels."""
    if count <= settings.observation_cap:
        return None
    return _ask(
        state,
        (
            f"В один ответ входит не больше {settings.observation_cap} наблюдений. "
            "Назовите меньше подписей."
        ),
        settings=settings,
    ) | {"label_reply": True}


def _question_plan(
    state: dict[str, Any],
    needles: list[str],
    periods: list[dict[str, str]],
) -> dict[str, Any]:
    question = str(state.get("question") or "")
    reply = str(state.get("human_reply") or "")
    folded = f"{question.casefold()} {reply.casefold()}"
    why = asks_how(folded)
    if len(needles) > 1:
        question_type = "compose"
    elif len(periods) >= 2 or "сравни" in folded:
        question_type = "compare"
    elif why:
        question_type = "explain"
    else:
        question_type = "lookup"
    return {
        "content_steps": state.get("content_steps", 0) + 1,
        "plan": {
            "question_type": question_type,
            "needles": needles,
            "periods": periods,
            "trace": "precedents" if why else "none",
            "source": "question",
        },
        "human_reply": "",
        "search_again": False,
        "draft_from_cache": False,
        "cache_missing": False,
        "terminal": "",
    }


def _phrase_tokens(needle: str, period_keys: set[str]) -> list[str]:
    """Words of a catalog phrase. A period key of this book is not a label."""
    tokens: list[str] = []
    for raw in needle.split():
        token = raw.strip(_TOKEN_EDGE)
        folded = token.casefold()
        if len(folded) < 2 or folded in period_keys:
            continue
        tokens.append(token)
    return tokens


async def _catalog_pages(
    parser: ParserClient,
    job_id: str,
    needles: list[Any],
    period_keys: set[str],
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Search each phrase once. A miss stays an empty page."""
    fetched: dict[str, tuple[str, dict[str, Any]]] = {}
    pages: list[tuple[str, dict[str, Any]]] = []
    added: set[str] = set()

    async def fetch(query: str) -> tuple[str, dict[str, Any]]:
        folded = query.casefold()
        cached = fetched.get(folded)
        if cached is not None:
            return cached
        page = await parser.search_rows(job_id, query, limit=limit)
        fetched[folded] = (query, page)
        return fetched[folded]

    def add(item: tuple[str, dict[str, Any]]) -> None:
        folded = item[0].casefold()
        if folded in added:
            return
        added.add(folded)
        pages.append(item)

    for raw in needles:
        tokens = _phrase_tokens(str(raw), period_keys)
        if not tokens:
            continue
        phrase = " ".join(tokens)
        add(await fetch(phrase))
    return pages


def _named_jobs(text: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Jobs whose filename or id is in the reply. A shared stem matches nobody."""
    folded = text.casefold()
    full: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in jobs:
        filename = str(job.get("source_filename") or "").strip()
        job_id = str(job.get("job_id") or "").strip()
        hit = bool(filename and filename.casefold() in folded)
        hit = hit or bool(job_id and job_id.casefold() in folded)
        if not hit:
            continue
        key = job_id or filename.casefold()
        if key in seen:
            continue
        seen.add(key)
        full.append(job)
    if full:
        return full
    prepared = [
        (job, set(file_segments(str(job.get("source_filename") or "")))) for job in jobs
    ]
    owners: list[dict[str, Any]] = []
    for token in name_tokens(folded):
        if len(token) < 3:
            continue
        matched = [job for job, segments in prepared if token in segments]
        if len(matched) != 1:
            continue
        job = matched[0]
        key = str(job.get("job_id") or "").strip()
        key = key or str(job.get("source_filename") or "").casefold()
        if key in seen:
            continue
        seen.add(key)
        owners.append(job)
    return owners


def _match_job(reply: str, jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    folded = reply.strip().casefold()
    if not folded:
        return None
    exact = [
        job
        for job in jobs
        if folded
        in {
            str(job.get("job_id") or "").casefold(),
            str(job.get("source_filename") or "").casefold(),
        }
    ]
    if len(exact) == 1:
        return exact[0]
    partial = [
        job
        for job in jobs
        if folded in str(job.get("source_filename") or "").casefold()
        or folded in str(job.get("job_id") or "").casefold()
    ]
    if len(partial) == 1:
        return partial[0]
    return None


async def _dependents(
    parser: ParserClient,
    state: dict[str, Any],
    selected: list[dict[str, Any]],
    period_ids: list[str],
    steps: list[dict[str, Any]],
    remaining: int,
    settings: Settings,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    origin_keys = {row["row_key"] for row in selected}
    for row in selected:
        traced = await parser.trace_dependents(
            state["job_id"], row["row_key"], depth=settings.precedent_depth
        )
        keys = []
        for node in traced.get("nodes") or []:
            key = node.get("row_key")
            if key and key not in origin_keys and key not in keys:
                keys.append(key)
        for key in keys[: settings.max_dependent_rows]:
            if remaining <= 0:
                return found
            document = await parser.get_observations(
                state["job_id"],
                row_key=key,
                period_ids=period_ids,
                precedent_depth=0,
                limit=min(settings.dependent_observation_limit, remaining),
            )
            batch = [
                without_account_code(scalar_observation(item))
                for item in (document.get("observations") or [])
            ]
            steps.append(
                {
                    "kind": "dependents",
                    "row_key": key,
                    "ids": [f"{item.get('row_key')}:{item.get('period_id')}" for item in batch],
                }
            )
            found.extend(batch)
            remaining -= len(batch)
    return found


def _accepted_citations(
    citations: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted = []
    for citation in citations:
        observation = next(
            (
                item
                for item in observations
                if item.get("row_key") == citation.get("row_key")
                and item.get("period_id") == citation.get("period_id")
            ),
            None,
        )
        if observation is None:
            continue
        accepted.append(
            {
                "row_key": citation.get("row_key"),
                "label": observation.get("label"),
                "period_id": citation.get("period_id"),
                "cell": (observation.get("source") or {}).get("cell"),
                "value": citation.get("value"),
                "value_status": citation.get("value_status") or observation.get("value_status"),
                "normalized_value": citation.get("normalized_value"),
            }
        )
    return accepted
