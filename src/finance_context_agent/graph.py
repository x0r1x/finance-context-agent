"""Fixed LangGraph. The model names needles and wording. Code fetches slices."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from finance_context_agent.citations import verify_answer, wants_influence
from finance_context_agent.llm import JsonModel, SchemaError
from finance_context_agent.parser import ParserClient, ParserError
from finance_context_agent.periods import periods_from_question, resolve_periods
from finance_context_agent.prompts import (
    answer_messages,
    critic_messages,
    plan_messages,
    without_account_code,
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
                        "ids": [
                            f"{item.get('row_key')}:{item.get('period_id')}" for item in batch
                        ],
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
        if scale_only or (not code_gaps and state.get("draft_from_cache")):
            return {
                "gaps": [],
                "gap_keys": [],
                "same_gap": False,
                "satisfactory": True,
                "citations": _accepted_citations(proposed, observed),
                "terminal": "",
            }
        if code_gaps:
            return {
                "gaps": code_gaps,
                "gap_keys": code_gaps,
                "same_gap": code_gaps == (state.get("gap_keys") or []),
                "satisfactory": False,
                "terminal": "",
            }
        system, user = critic_messages({**state, "gaps": []})
        try:
            parsed = await model.complete_json(role="critic", system=system, user=user)
        except SchemaError:
            return {
                "gaps": ["schema"],
                "gap_keys": ["schema"],
                "same_gap": True,
                "satisfactory": False,
                "terminal": "",
            }
        critic_gaps = _actionable_critic_gaps(list(parsed.get("gaps") or []))
        return {
            "gaps": critic_gaps,
            "gap_keys": critic_gaps,
            "same_gap": bool(critic_gaps) and critic_gaps == (state.get("gap_keys") or []),
            "satisfactory": not critic_gaps,
            "citations": _accepted_citations(
                list(state.get("proposed_citations") or []),
                list(state.get("observations") or []),
            )
            if not critic_gaps
            else state.get("citations") or [],
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
    pending = [
        (needle, page) for needle, page in pages if int(page.get("total") or 0) != 1
    ]
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


_CRITIC_EXACT = {"compare_sides", "precedent", "schema"}


def _actionable_critic_gaps(items: list[Any]) -> list[str]:
    """Keep critic notes the code can replan. A free-text nag must not hide a checked citation."""
    kept: list[str] = []
    for item in items:
        gap = str(item).strip()
        if gap in _CRITIC_EXACT or gap.startswith(("cell:", "number:", "scale:")):
            kept.append(gap)
    return kept


_MENTION_STOP = frozenset(
    {
        "какой",
        "какая",
        "какое",
        "какие",
        "почему",
        "равен",
        "равна",
        "равно",
        "сравни",
        "в",
        "и",
        "на",
        "за",
        "по",
        "год",
        "года",
        "году",
        "покажи",
        "покажите",
        "первую",
        "первой",
        "первый",
        "первая",
        "первое",
        "операционный",
        "операционного",
        "операционном",
        "начала",
        "начало",
        "погашения",
    }
)
_MENTION_EDGE = "?.!,;:«»\"'"
_WHY_PHRASES = ("почему", "из чего", "why")
_STATUS_WORDS = {"empty": "пусто", "not_applicable": "не применимо"}


def question_mention(question: str, axes: list[dict[str, Any]]) -> str:
    """Label text left after period keys, function words, and a bare number."""
    period_keys = {
        str(period.get("period_key") or "").casefold()
        for axis in axes
        for period in axis.get("periods") or []
        if period.get("period_key")
    }
    tokens: list[str] = []
    for raw in question.split():
        token = raw.strip(_MENTION_EDGE)
        folded = token.casefold()
        if len(folded) < 2 or folded in _MENTION_STOP or folded in period_keys:
            continue
        if folded.isdigit():
            continue
        tokens.append(token)
    return " ".join(tokens)


_COVER_PHRASES = ("ключевые показатели", "общая картина", "обложка")


def question_needles(question: str, axes: list[dict[str, Any]]) -> list[str]:
    """One needle per label. A conjunction between periods stays one needle."""
    text, _hit = _strip_cover(question)
    needles: list[str] = []
    for span in _label_spans(text, axes):
        mention = question_mention(span, axes)
        if mention:
            needles.append(mention)
    return needles


def _strip_cover(question: str) -> tuple[str, bool]:
    text = question
    hit = False
    for phrase in _COVER_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(text):
            hit = True
            text = pattern.sub(" ", text)
    return text, hit


def _is_cover(question: str) -> bool:
    folded = question.casefold()
    return any(phrase in folded for phrase in _COVER_PHRASES)


def _label_spans(question: str, axes: list[dict[str, Any]]) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"\s*,\s*", question) if piece.strip()]
    spans: list[str] = []
    for piece in pieces:
        spans.extend(_split_and(piece, axes))
    return spans


def _split_and(piece: str, axes: list[dict[str, Any]]) -> list[str]:
    match = re.search(r"\s+и\s+", piece, flags=re.IGNORECASE)
    if match is None:
        return [piece]
    left, right = piece[: match.start()], piece[match.end() :]
    if question_mention(left, axes) and question_mention(right, axes):
        return _split_and(left, axes) + _split_and(right, axes)
    return [piece]


def _cover_question(summary: dict[str, Any]) -> str:
    names = [
        str(sheet)
        for sheet in (summary.get("sheets") or [])
        if str(sheet) and not any(char.isdigit() for char in str(sheet))
    ]
    lines = ["Какую строку открыть?"]
    if len(names) >= 2:
        lines.append("Листы: " + ", ".join(names) + ".")
    lines.append(
        "Назовите до четырёх: CFADS, обслуживание долга, DSCR, Project IRR, Equity IRR."
    )
    return "\n".join(lines)


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
    why = any(phrase in folded for phrase in _WHY_PHRASES)
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


_ACRONYM = re.compile(r"\(([A-Za-z]{2,12})\)\s*$")
_KIND_WORDS = {
    "abstract": "заголовок",
    "fact": "значение",
    "helper": "расчёт",
    "flag": "флаг",
    "params": "параметр",
}


def narrow_hits(mention: str, page: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep a row by its own label. A trailing acronym is the same line."""
    folded = mention.casefold().strip()
    rows = _on_own_label(folded, list(page.get("rows") or []))
    if not folded:
        return rows
    exact = [row for row in rows if str(row.get("label") or "").casefold() == folded]
    if not exact:
        return rows
    longer = [
        row
        for row in rows
        if row not in exact and _keeps_longer(str(row.get("label") or ""), folded)
    ]
    return exact + longer


def _on_own_label(needle: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not needle:
        return rows
    return [row for row in rows if needle in str(row.get("label") or "").casefold()]


def _keeps_longer(label: str, needle: str) -> bool:
    found = _ACRONYM.search(label.strip())
    if found and found.group(1).casefold() == needle:
        return True
    folded = label.casefold()
    if not folded.startswith(needle):
        return False
    return len(folded) == len(needle) or not folded[len(needle)].isalnum()


def _narrow_page(mention: str, page: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rows = list(page.get("rows") or [])
    total = int(page.get("total") or 0)
    kept = narrow_hits(mention, page)
    if len(kept) == len(rows):
        return mention, page
    truncated = total > len(rows)
    if truncated and len(kept) <= 1:
        return mention, {**page, "rows": kept, "total": total}
    return mention, {**page, "rows": kept, "total": len(kept)}


def _cache_answer(state: dict[str, Any]) -> dict[str, Any]:
    selected = list(state.get("selected") or [])
    period_ids = [str(item) for item in (state.get("period_ids") or [])]
    wanted = {row.get("row_key") for row in selected}
    scoped = [
        item
        for item in (state.get("observations") or [])
        if item.get("row_key") in wanted
        and (not period_ids or str(item.get("period_id") or "") in period_ids)
    ]
    if period_ids:
        found = {(item.get("row_key"), str(item.get("period_id") or "")) for item in scoped}
        missing = any(
            (row.get("row_key"), period_id) not in found
            for row in selected
            for period_id in period_ids
        )
        if missing:
            return {
                "draft": "Подтверждённого числа в срезе нет.",
                "citations": [],
                "missing": True,
            }
        ordered = []
        for row in selected:
            for period_id in period_ids:
                ordered.append(
                    next(
                        item
                        for item in scoped
                        if item.get("row_key") == row.get("row_key")
                        and str(item.get("period_id") or "") == period_id
                    )
                )
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in scoped:
            grouped.setdefault(str(item.get("row_key")), []).append(item)
        one_each = len(grouped) == len(selected) and all(
            len(items) == 1 for items in grouped.values()
        )
        if not one_each:
            return {
                "draft": "Подтверждённого числа в срезе нет.",
                "citations": [],
                "missing": True,
            }
        ordered = [grouped[str(row.get("row_key"))][0] for row in selected]
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for item in ordered:
        line, citation = _line_from_observation(item)
        lines.append(line)
        citations.append(citation)
    return {"draft": "\n".join(lines), "citations": citations, "missing": False}


def _scalar_observation(item: dict[str, Any]) -> dict[str, Any]:
    """A params column keyed ``value`` is the scalar slot, not a period on an axis."""
    period = str(item.get("period_id") or "")
    if period.casefold() != "value":
        return item
    return {**item, "period_id": ""}


def _line_from_observation(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    item = _scalar_observation(item)
    status = str(item.get("value_status") or "")
    label = str(item.get("label") or item.get("row_key") or "")
    period_id = str(item.get("period_id") or "")
    if status in _STATUS_WORDS:
        text_value = _STATUS_WORDS[status]
        cited_value = ""
    else:
        cited_value = "" if item.get("value") is None else str(item.get("value"))
        text_value = cited_value
        if status != "zero_explicit" and _needs_scale_word(item):
            text_value = f"{text_value} тыс."
    if period_id:
        line = f"{label} в {period_id}: {text_value}"
    else:
        line = f"{label}: {text_value}"
    citation: dict[str, Any] = {
        "row_key": item.get("row_key"),
        "period_id": period_id,
        "cell": (item.get("source") or {}).get("cell") or "",
        "value": cited_value,
        "value_status": status,
    }
    normalized = item.get("normalized_value")
    if normalized not in (None, ""):
        citation["normalized_value"] = normalized
    return line, citation


def _needs_scale_word(item: dict[str, Any]) -> bool:
    factor = item.get("scale_factor")
    if factor in (None, 1):
        return False
    scale = ((item.get("unit") or {}).get("scale") or "").casefold()
    return scale == "k"


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


def _choice_question(pages: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for needle, page in pages:
        total = int(page.get("total") or 0)
        rows = page.get("rows") or []
        labels = _menu_labels(rows)
        line = f"«{needle}»: {total}. {labels}"
        if total > len(rows):
            line += f" Показаны первые {len(rows)} из {total}."
        lines.append(line)
    return "Какую строку взять?\n" + "\n".join(lines)


def _menu_labels(rows: list[dict[str, Any]]) -> str:
    base = [_label(row) for row in rows]
    counts: dict[str, int] = {}
    for text in base:
        counts[text] = counts.get(text, 0) + 1
    shown: list[str] = []
    for row, text in zip(rows, base, strict=True):
        if counts[text] > 1:
            word = _KIND_WORDS.get(str(row.get("kind") or ""))
            if word:
                text = f"{text}, {word}"
        shown.append(text)
    return "; ".join(shown)


def _label(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "")
    sheet = str(row.get("sheet") or "")
    heading = _heading(row.get("label_path"), label)
    mark = ", ".join(part for part in (sheet, heading) if part)
    if not mark:
        return label
    return f"{label} [{mark}]"


def _heading(path: Any, label: str) -> str:
    if not isinstance(path, list):
        return ""
    folded = label.casefold()
    for part in reversed(path):
        text = str(part).strip()
        if not text or text.casefold() == folded or any(char.isdigit() for char in text):
            continue
        return text
    return ""


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row.get("row_key"),
        "label": row.get("label"),
        "sheet": row.get("sheet"),
        "label_path": list(row.get("label_path") or []),
        "axis_ids": list(row.get("axis_ids") or []),
        "disposition": row.get("disposition"),
        "kind": row.get("kind"),
    }


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
                _scalar_observation(item) for item in (document.get("observations") or [])
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
