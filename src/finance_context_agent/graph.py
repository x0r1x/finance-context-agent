"""Fixed LangGraph. Code names needles from the question and fetches slices."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from finance_context_agent.catalog import _choice_question, _compact_row, _narrow_page
from finance_context_agent.citations import verify_answer, wants_influence
from finance_context_agent.draft import _cache_answer, _scalar_observation
from finance_context_agent.llm import JsonModel, SchemaError
from finance_context_agent.parser import ParserClient, ParserError
from finance_context_agent.periods import periods_from_question, resolve_periods
from finance_context_agent.prompts import (
    answer_messages,
    plan_messages,
    without_account_code,
)
from finance_context_agent.questions import (
    _cover_question,
    _is_cover,
    asks_how,
    question_needles,
)
from finance_context_agent.settings import Settings

_TOKEN_EDGE = "?.!,;:«»\"'[]"


class AgentState(TypedDict, total=False):
    question: str
    job_id: str
    jobs: list[dict[str, Any]]
    pending: str
    awaiting: str
    user_question: str
    reload: bool
    just_bound_job: bool
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
    label_reply: bool
    clarify_rounds: int
    schema_error: bool
    draft_from_cache: bool
    cache_missing: bool


def build_graph(
    parser: ParserClient,
    model: JsonModel,
    checkpointer: Any,
    settings: Settings | None = None,
) -> Any:
    settings = settings or Settings()

    async def load(state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("job_id"):
            try:
                jobs = await parser.list_jobs(status="succeeded")
            except ParserError as exc:
                return _terminal_or_raise(exc)
            if not jobs:
                return _done("Готовых книг нет.", ["no_jobs"])
            lines = [f"- {job.get('source_filename') or job.get('job_id')}" for job in jobs]
            return {
                "jobs": jobs,
                "pending": "job",
                "awaiting": "job",
                "user_question": "Какую книгу открыть?\n" + "\n".join(lines),
                "reload": False,
                "just_bound_job": False,
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
        question = str(state.get("question") or "")
        axes = list(state.get("axes") or [])
        reply = str(state.get("human_reply") or "").strip()
        text = reply or question
        needles = question_needles(text, axes)
        named = periods_from_question(text, axes)
        if reply and not named:
            named = periods_from_question(question, axes)
        if needles and (not reply or state.get("label_reply")):
            if len(needles) > settings.max_needles:
                return _ask(state, "Назовите не больше четырёх.", settings=settings) | {
                    "label_reply": True
                }
            return _question_plan(state, needles, named, settings) | {"label_reply": False}
        if not reply and _is_cover(question):
            return _ask(state, _cover_question(state.get("summary") or {}), settings=settings) | {
                "label_reply": True
            }
        system, user = plan_messages(state)
        try:
            parsed = await model.complete_json(role="plan", system=system, user=user)
        except SchemaError:
            return _done("Модель не вернула план.", ["schema"])
        model_needles = [
            str(item).strip() for item in parsed.get("needles") or [] if str(item).strip()
        ]
        periods = named or list(parsed.get("periods") or [])
        return {
            "content_steps": state.get("content_steps", 0) + 1,
            "plan": {
                "question_type": parsed.get("question_type") or "lookup",
                "needles": model_needles[: settings.max_needles],
                "periods": periods,
                "trace": parsed.get("trace") or "none",
                "source": "model",
            },
            "human_reply": "",
            "search_again": False,
            "draft_from_cache": False,
            "cache_missing": False,
            "label_reply": False,
            "terminal": "",
        }

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
        pages = [_narrow_page(needle, page) for needle, page in pages]
        if not pages:
            return _miss(state, "Пустой поиск.", settings)
        if (plan_body.get("question_type") or "lookup") == "compose":
            return _compose(state, pages, settings)
        return _single(state, pages, settings)

    async def bind(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("pending") != "job":
            return {"pending": "", "terminal": ""}
        found = _match_job(state.get("human_reply") or "", state.get("jobs") or [])
        if found is None:
            if state.get("clarify_rounds", 0) >= settings.clarify_budget:
                return _done("Книга не выбрана.", ["job"])
            return {
                "user_question": "Не нашёл такую книгу. Назовите файл ещё раз.",
                "awaiting": "job",
                "pending": "job",
                "terminal": "",
            }
        return {
            "job_id": found["job_id"],
            "pending": "",
            "awaiting": "",
            "human_reply": "",
            "user_question": "",
            "just_bound_job": True,
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
                    without_account_code(_scalar_observation(item))
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
            if (
                plan_body.get("source") == "question"
                and not period_ids
                and _series_without_period(observations)
            ):
                return _ask(state, "Назовите период.", steps=steps, settings=settings)
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
            built = _cache_answer(state)
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
        {"search": "search", "ask_user": "ask_user", "close": "close"},
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
        {"ask_user": "ask_user", "load": "load", "plan": "plan", "close": "close"},
    )
    builder.add_conditional_edges(
        "retrieve",
        _route_retrieve,
        {"ask_user": "ask_user", "answer": "answer", "close": "close"},
    )
    builder.add_edge("answer", "check")

    def route_check(state: dict[str, Any]) -> str:
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
    if state.get("awaiting") == "job":
        return "ask_user"
    if state.get("just_bound_job"):
        return "load"
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


def _miss(state: dict[str, Any], note: str, settings: Settings) -> dict[str, Any]:
    misses = state.get("search_misses", 0) + 1
    if misses >= 2:
        return _ask(
            state,
            "Такой строки нет. Назовите подпись иначе или выберите другую метрику.",
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
        return _ask(state, _choice_question(useful), settings=settings)
    return _select(state, rows, settings)


def _compose(
    state: dict[str, Any], pages: list[tuple[str, dict[str, Any]]], settings: Settings
) -> dict[str, Any]:
    if all(int(page.get("total") or 0) == 0 for _needle, page in pages):
        return _miss(state, "Нет совпадений по метрикам.", settings)
    pending = [(needle, page) for needle, page in pages if int(page.get("total") or 0) != 1]
    if pending:
        return _ask(state, _choice_question(pending), settings=settings)
    return _select(state, _distinct_rows(pages), settings)


def _select(
    state: dict[str, Any], rows: list[dict[str, Any]], settings: Settings
) -> dict[str, Any]:
    plan_body = state.get("plan") or {}
    requested = list(plan_body.get("periods") or [])
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
    selected = [_compact_row(row) for row in rows]
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


def _series_without_period(observations: list[dict[str, Any]]) -> bool:
    counts: dict[str, int] = {}
    for item in observations:
        key = str(item.get("row_key"))
        counts[key] = counts.get(key, 0) + 1
    return any(count > 1 for count in counts.values())


def _question_plan(
    state: dict[str, Any],
    needles: list[str],
    periods: list[dict[str, str]],
    settings: Settings,
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
            "needles": needles[: settings.max_needles],
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
    """Search a phrase once. A total miss with several tokens searches each token."""
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
        phrase_item = await fetch(phrase)
        if int(phrase_item[1].get("total") or 0) > 0 or len(tokens) == 1:
            add(phrase_item)
            continue
        token_pages = [await fetch(token) for token in tokens]
        if any(int(page.get("total") or 0) > 0 for _token, page in token_pages):
            for item in token_pages:
                add(item)
            continue
        add(phrase_item)
    return pages


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
                without_account_code(_scalar_observation(item))
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
