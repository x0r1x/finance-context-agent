#!/usr/bin/env python3
"""Score business answers against the parser observation cache.

The oracle is the cached cell, compared as text. Decimal folding only accepts
grouping spaces and a decimal comma. It is not a binary float compare.
A catalog total in a menu (``«EBITDA»: 2.``) is not a cell value.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?%?(?!\w)")
_MENU_COUNT = re.compile(r"»: \d+\.|Показаны первые \d+ из \d+\.")
_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,255}$")

_SCALE_WORDS = {
    "k": ("k", "тыс", "thousand"),
    "m": ("m", "млн", "million"),
    "bn": ("bn", "млрд", "billion"),
}

_BOOKS = {"packt-project-finance.xlsx", "rvi-project-finance.xlsx"}
_GROUPS = {"point", "compare", "calendar", "scalar", "menu", "phase", "composition"}
_OUTCOMES = {
    "citation": ("row_key", "period_id"),
    "citations": ("pairs",),
    "ask": (),
    "refuse": (),
    "citation_or_ask": ("allowed",),
    "any_row_or_ask": ("period_id",),
}

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CORPUS = "/Users/alekseykashin/projects/finance-context-builder/resources/corpus"


def cases_path() -> Path:
    return Path(__file__).resolve().parent / "business_cases.json"


def load_cases(path: Path | None = None) -> dict[str, Any]:
    document = json.loads((path or cases_path()).read_text(encoding="utf-8"))
    _validate(document)
    return document


def score_case(
    case: dict[str, Any],
    completion: dict[str, Any],
    observation_for: Any,
) -> dict[str, Any]:
    """Return attribute, value, and wording flags. Green is attribute and value."""
    expect = case["expect"]
    content = _content(completion)
    citations = _citations(completion)
    gaps = [str(item) for item in (completion.get("gaps") or [])]
    steps = list(completion.get("steps") or [])
    awaiting = bool(completion.get("awaiting_user"))
    scan = content if citations else _MENU_COUNT.sub(" ", _mask_menu_labels(content))
    allowed = _allowed_tokens(citations, observation_for) if citations else set()
    strays = _stray_numbers(scan, allowed)
    attribute_ok, value_ok, notes = _score_outcome(
        expect, content, citations, awaiting, strays, observation_for
    )
    scale_notes = _scale_notes(content, citations, observation_for)
    wording_notes = [_number_note(strays), *scale_notes]
    wording_notes = [item for item in wording_notes if item]
    precedent = _saw_precedent(gaps, steps)
    return {
        "id": case["id"],
        "group": case.get("group") or "",
        "book": case.get("book") or "",
        "question": case.get("question") or "",
        "expect": expect,
        "attribute_ok": attribute_ok,
        "value_ok": value_ok,
        "wording_ok": not wording_notes,
        "passed": attribute_ok and value_ok,
        "awaiting": awaiting,
        "content": content,
        "citations": citations,
        "gaps": gaps,
        "precedent": precedent,
        "note": "; ".join([*notes, *wording_notes]),
    }


def _score_outcome(
    expect: dict[str, Any],
    content: str,
    citations: list[dict[str, Any]],
    awaiting: bool,
    strays: list[str],
    observation_for: Any,
) -> tuple[bool, bool, list[str]]:
    outcome = expect["outcome"]
    if outcome == "citation":
        return _score_citation(expect, citations, observation_for)
    if outcome == "citations":
        return _score_pairs(expect, citations, observation_for)
    if outcome == "ask":
        return _score_ask(expect, content, citations, awaiting, strays)
    if outcome == "refuse":
        return _score_refuse(citations, strays)
    if outcome == "citation_or_ask":
        return _score_citation_or_ask(expect, citations, strays, observation_for)
    if outcome == "any_row_or_ask":
        return _score_any_row(expect, citations, strays, observation_for)
    return False, False, [f"неизвестный исход {outcome}"]


def _score_citation(
    expect: dict[str, Any],
    citations: list[dict[str, Any]],
    observation_for: Any,
) -> tuple[bool, bool, list[str]]:
    target = (str(expect["row_key"]), str(expect.get("period_id") or ""))
    notes = _forbid_notes(expect, citations)
    got = [_pair(item) for item in citations]
    attribute_ok = got == [target] and not notes
    if not attribute_ok:
        notes.append(_citation_note(got, target))
        return False, False, notes
    ok, note = _value_matches(citations[0], observation_for(*target), _status(expect))
    return True, ok, [note] if note else []


def _score_pairs(
    expect: dict[str, Any],
    citations: list[dict[str, Any]],
    observation_for: Any,
) -> tuple[bool, bool, list[str]]:
    expected = [_pair(item) for item in expect["pairs"]]
    got = [_pair(item) for item in citations]
    if sorted(got) != sorted(expected) or len(got) != len(expected):
        return False, False, ["нужны все указанные точки и никаких других"]
    by_pair = {_pair(item): item for item in citations}
    notes: list[str] = []
    value_ok = True
    for item in expect["pairs"]:
        pair = _pair(item)
        ok, note = _value_matches(by_pair[pair], observation_for(*pair), _status(item))
        if not ok:
            value_ok = False
            notes.append(f"{pair[1] or 'скаляр'}: {note}")
    return True, value_ok, notes


def _score_ask(
    expect: dict[str, Any],
    content: str,
    citations: list[dict[str, Any]],
    awaiting: bool,
    strays: list[str],
) -> tuple[bool, bool, list[str]]:
    folded = content.casefold()
    missing = [
        piece for piece in expect.get("menu_contains") or [] if str(piece).casefold() not in folded
    ]
    notes: list[str] = []
    if not awaiting:
        notes.append("нет паузы")
    if citations:
        notes.append("на паузе есть цитата")
    if missing:
        notes.append("в меню нет: " + "; ".join(missing))
    attribute_ok = awaiting and not citations and not missing
    value_ok = not citations and not strays
    return attribute_ok, value_ok, notes


def _score_refuse(
    citations: list[dict[str, Any]],
    strays: list[str],
) -> tuple[bool, bool, list[str]]:
    notes: list[str] = []
    if citations:
        notes.append("вместо отказа есть цитата")
    return not citations, not citations and not strays, notes


def _score_citation_or_ask(
    expect: dict[str, Any],
    citations: list[dict[str, Any]],
    strays: list[str],
    observation_for: Any,
) -> tuple[bool, bool, list[str]]:
    if not citations:
        return True, not strays, []
    allowed = {_pair(item): item for item in expect["allowed"]}
    if len(citations) != 1 or _pair(citations[0]) not in allowed:
        return False, False, ["цитата не из разрешённого списка"]
    chosen = allowed[_pair(citations[0])]
    ok, note = _value_matches(citations[0], observation_for(*_pair(chosen)), _status(chosen))
    return True, ok, [note] if note else []


def _score_any_row(
    expect: dict[str, Any],
    citations: list[dict[str, Any]],
    strays: list[str],
    observation_for: Any,
) -> tuple[bool, bool, list[str]]:
    if not citations:
        return True, not strays, []
    period = str(expect.get("period_id") or "")
    if len(citations) != 1:
        return False, False, ["нужна одна строка или пауза без числа"]
    citation = citations[0]
    if _period(citation) != period:
        return False, False, [f"период {_period(citation) or '—'} вместо {period}"]
    observation = observation_for(_row(citation), period)
    if observation is None:
        return False, False, ["кэш не найден"]
    needle = str(expect.get("label_contains") or "")
    label = str(observation.get("label") or "")
    if needle and needle.casefold() not in label.casefold():
        return False, False, [f"подпись «{label}» без «{needle}»"]
    ok, note = _value_matches(citation, observation, _status(expect))
    return True, ok, [note] if note else []


def _value_matches(
    citation: dict[str, Any],
    observation: dict[str, Any] | None,
    expected_status: str | None,
) -> tuple[bool, str]:
    if observation is None:
        return False, "кэш не найден"
    cited_status = str(citation.get("value_status") or "")
    observed_status = str(observation.get("value_status") or "")
    cited = citation.get("value")
    cited_text = "" if cited is None else str(cited).strip()
    if expected_status and observed_status != expected_status:
        return False, f"в кэше {observed_status or '—'}, ждали {expected_status}"
    if observed_status in {"empty", "not_applicable"}:
        if cited_text:
            return False, "пустой кэш назван числом"
        if cited_status != observed_status:
            return False, f"статус {cited_status or '—'} вместо {observed_status}"
        return True, ""
    if observed_status == "zero_explicit":
        if cited_text != "0" or cited_status != "zero_explicit":
            return False, "явный ноль должен остаться 0"
        return True, ""
    if not cited_text:
        return False, "в цитате нет значения"
    if not (
        _same_text(cited_text, observation.get("value"))
        or _same_text(cited_text, observation.get("normalized_value"))
    ):
        return False, "значение не равно кэшу"
    if cited_status != observed_status:
        return False, f"статус {cited_status or '—'} вместо {observed_status}"
    return True, ""


def _scale_notes(
    content: str,
    citations: list[dict[str, Any]],
    observation_for: Any,
) -> list[str]:
    notes: list[str] = []
    for citation in citations:
        observation = observation_for(_row(citation), _period(citation))
        if observation is None or not _scale_required(citation, observation):
            continue
        words = _scale_words(observation)
        if words and not _has_scale(content, words):
            notes.append(f"не назван масштаб {words[0]}")
    return notes


def _scale_required(citation: dict[str, Any], observation: dict[str, Any]) -> bool:
    if _factor_is_one(observation):
        return False
    status = citation.get("value_status") or observation.get("value_status")
    if status == "zero_explicit" and str(citation.get("value") or "").strip() == "0":
        return False
    normalized = observation.get("normalized_value")
    display = observation.get("value")
    if _same_text(citation.get("value"), normalized) and not _same_text(
        citation.get("value"), display
    ):
        return False
    return bool(_scale_words(observation))


def _factor_is_one(observation: dict[str, Any]) -> bool:
    factor = observation.get("scale_factor")
    if factor in (None, 1):
        return True
    return _fold(factor) == "1"


def _scale_words(observation: dict[str, Any]) -> tuple[str, ...]:
    unit = observation.get("unit") or {}
    scale = ""
    if isinstance(unit, dict):
        scale = str(unit.get("scale") or "").casefold()
    if scale in _SCALE_WORDS:
        return _SCALE_WORDS[scale]
    return (scale,) if scale else ()


def _has_scale(text: str, words: tuple[str, ...]) -> bool:
    folded = text.casefold()
    for word in words:
        if not word:
            continue
        if re.search(rf"(?<![^\W\d_]){re.escape(word)}(?!\w)", folded):
            return True
        if len(word) >= 3 and re.search(rf"(?<![^\W\d_]){re.escape(word)}", folded):
            return True
    return False


def _allowed_tokens(
    citations: list[dict[str, Any]],
    observation_for: Any,
) -> set[str]:
    allowed: set[str] = set()
    for citation in citations:
        raws: list[Any] = [citation.get("value"), citation.get("period_id")]
        observation = observation_for(_row(citation), _period(citation))
        if observation:
            raws.append(observation.get("normalized_value"))
            formula = observation.get("formula") or {}
            if isinstance(formula, dict):
                raws.extend(_NUMBER.findall(str(formula.get("text") or "")))
                for precedent in formula.get("precedents") or []:
                    if not isinstance(precedent, dict):
                        continue
                    raws.append(precedent.get("value"))
                    raws.append(precedent.get("period_id"))
        for raw in raws:
            _remember(allowed, raw)
    return allowed


def _remember(allowed: set[str], raw: Any) -> None:
    if raw is None:
        return
    text = str(raw).strip()
    if not text:
        return
    allowed.add(text)
    folded = _fold(text)
    if folded is not None:
        allowed.add(folded)


def _stray_numbers(text: str, allowed: set[str]) -> list[str]:
    found: list[str] = []
    for token in _NUMBER.findall(text):
        stripped = token.strip()
        folded = _fold(stripped)
        if folded is not None and folded in allowed:
            continue
        if stripped in allowed:
            continue
        found.append(stripped)
    return found


def _number_note(strays: list[str]) -> str:
    if not strays:
        return ""
    return "в тексте есть число вне цитаты: " + ", ".join(strays)


def _saw_precedent(gaps: list[str], steps: list[Any]) -> bool:
    if any(gap == "precedent" or gap.startswith("precedent") for gap in gaps):
        return True
    for step in steps:
        if not isinstance(step, dict):
            continue
        try:
            depth = int(step.get("precedent_depth") or 0)
        except (TypeError, ValueError):
            depth = 0
        if depth > 0:
            return True
    return False


def _forbid_notes(expect: dict[str, Any], citations: list[dict[str, Any]]) -> list[str]:
    forbidden = {str(item) for item in expect.get("forbid_periods") or []}
    notes: list[str] = []
    for citation in citations:
        period = _period(citation)
        if period in forbidden:
            notes.append(f"запрещённый период {period}")
    return notes


def _citation_note(got: list[tuple[str, str]], target: tuple[str, str]) -> str:
    if not got:
        return "цитаты нет"
    shown = ", ".join(f"{row} {period or 'скаляр'}" for row, period in got)
    return f"цитата {shown}, цель {target[0]} {target[1] or 'скаляр'}"


def _mask_menu_labels(content: str) -> str:
    """Drop menu labels. A digit inside ``last 12 months`` is not a cell value."""
    lines: list[str] = []
    for line in content.splitlines():
        marker = "»: "
        if marker not in line:
            lines.append(line)
            continue
        head, tail = line.split(marker, 1)
        dot = tail.find(".")
        if dot < 0:
            lines.append(line)
            continue
        kept: list[str] = []
        for part in tail[dot + 1 :].split("; "):
            bracket = part.rfind("]")
            if bracket == -1:
                kept.append(part)
                continue
            rest = part[bracket + 1 :]
            if rest.strip():
                kept.append(rest)
        lines.append(f"{head}{marker}{tail[: dot + 1]}{''.join(kept)}")
    return "\n".join(lines)


def _content(completion: dict[str, Any]) -> str:
    choices = completion.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


def _citations(completion: dict[str, Any]) -> list[dict[str, Any]]:
    raw = completion.get("citations") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _pair(item: dict[str, Any]) -> tuple[str, str]:
    return _row(item), _period(item)


def _row(item: dict[str, Any]) -> str:
    return str(item.get("row_key") or "")


def _period(item: dict[str, Any]) -> str:
    value = item.get("period_id")
    if value is None:
        return ""
    return str(value).strip()


def _status(item: dict[str, Any]) -> str | None:
    value = item.get("value_status")
    if value in (None, ""):
        return None
    return str(value)


def _same_text(left: Any, right: Any) -> bool:
    left_text = "" if left is None else str(left).strip()
    right_text = "" if right is None else str(right).strip()
    if left_text == right_text:
        return True
    folded_left = _fold(left_text)
    folded_right = _fold(right_text)
    return folded_left is not None and folded_left == folded_right


def _fold(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if text.endswith("%"):
        text = text[:-1]
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None
    return format(decimal.normalize(), "f")


def _validate(document: dict[str, Any]) -> None:
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = str(case.get("id") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"duplicate or empty id: {case_id}")
        seen.add(case_id)
        if case.get("book") not in _BOOKS:
            raise ValueError(f"{case_id}: unknown book")
        if case.get("group") not in _GROUPS:
            raise ValueError(f"{case_id}: unknown group")
        if not str(case.get("question") or "").strip():
            raise ValueError(f"{case_id}: empty question")
        expect = case.get("expect")
        if not isinstance(expect, dict):
            raise ValueError(f"{case_id}: expect must be an object")
        outcome = expect.get("outcome")
        if outcome not in _OUTCOMES:
            raise ValueError(f"{case_id}: unknown outcome")
        for key in _OUTCOMES[outcome]:
            if key not in expect:
                raise ValueError(f"{case_id}: missing {key}")
        _validate_points(case_id, expect)


def _validate_points(case_id: str, expect: dict[str, Any]) -> None:
    for key in ("pairs", "allowed"):
        items = expect.get(key)
        if items is None:
            continue
        if not isinstance(items, list) or not items:
            raise ValueError(f"{case_id}: {key} must be a non-empty list")
        for item in items:
            if not isinstance(item, dict) or "row_key" not in item or "period_id" not in item:
                raise ValueError(f"{case_id}: {key} item needs row_key and period_id")


def main() -> int:
    document = load_cases()
    agent = _base("AGENT_BASE_URL", "BASE_URL", "http://127.0.0.1:8090")
    parser = _base("PARSER_BASE_URL", "", "http://127.0.0.1:8080")
    chat_timeout = _env_float("CHAT_TIMEOUT_SEC", 180)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    thread_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_root = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else ROOT / "out"
    run_dir = out_root / "business-answers" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(os.environ.get("CORPUS_DIR", _DEFAULT_CORPUS))
    ready_status, ready_body = _request_json("GET", f"{agent}/readyz", None, 10)
    ready = (
        ready_status == 200
        and isinstance(ready_body, dict)
        and ready_body.get("status") == "ready"
        and ready_body.get("redis") is True
        and ready_body.get("parser") is True
    )
    jobs = _jobs(parser) if ready else {}
    results: list[dict[str, Any]] = []
    if ready:
        results = _run_cases(document["cases"], jobs, agent, parser, chat_timeout, thread_stamp)
    report = {
        "started": stamp,
        "agent": agent,
        "parser": parser,
        "ready": ready,
        "jobs": jobs,
        "corpus": _corpus_note(corpus, document),
        "skipped": list(document.get("skipped") or []),
        "cases": results,
    }
    _write_report(run_dir, report)
    print(str(run_dir), file=sys.stderr)
    if not ready:
        print("readyz is not ready", file=sys.stderr)
        return 1
    if any(not item["passed"] or item.get("http_status") != 200 for item in results):
        return 1
    return 0


def _run_cases(
    cases: list[dict[str, Any]],
    jobs: dict[str, str],
    agent: str,
    parser: str,
    chat_timeout: float,
    thread_stamp: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        job_id = jobs.get(str(case["book"]))
        if not job_id:
            result = _empty_result(case, "нет готовой задачи для этой книги")
            result["http_status"] = 0
            result["seconds"] = 0
            results.append(result)
            _progress(result)
            continue
        thread_id = f"biz-{case['id']}-{thread_stamp}"
        if not _THREAD_ID.fullmatch(thread_id):
            result = _empty_result(case, "thread_id не проходит проверку")
            result["http_status"] = 0
            result["seconds"] = 0
            results.append(result)
            _progress(result)
            continue
        started = datetime.now()
        status, body = _request_json(
            "POST",
            f"{agent}/v1/chat/completions",
            {
                "job_id": job_id,
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": case["question"]}],
            },
            chat_timeout,
        )
        seconds = (datetime.now() - started).total_seconds()
        completion = body if isinstance(body, dict) else {}
        if status == 200:
            result = score_case(case, completion, _lookup(parser, job_id))
        else:
            code = ""
            if isinstance(body, dict):
                code = str(body.get("error") or "")
            result = _empty_result(case, f"HTTP {status} {code}".strip())
        result["http_status"] = status
        result["seconds"] = round(seconds, 2)
        result["thread_id"] = thread_id
        result["job_id"] = job_id
        results.append(result)
        _progress(result)
    return results


def _lookup(parser: str, job_id: str) -> Any:
    cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    def observation_for(row_key: str, period_id: str) -> dict[str, Any] | None:
        key = (row_key, period_id or "")
        if key not in cache:
            cache[key] = _observation(parser, job_id, row_key, period_id or "")
        return cache[key]

    return observation_for


def _observation(parser: str, job_id: str, row_key: str, period_id: str) -> dict[str, Any] | None:
    if not row_key:
        return None
    params = [("row_key", row_key), ("limit", "8"), ("precedent_depth", "2")]
    if period_id:
        params.append(("period_id", period_id))
    query = urllib.parse.urlencode(params)
    url = f"{parser}/v1/context-jobs/{job_id}/observations?{query}"
    status, body = _request_json("GET", url, None, 30)
    if status != 200 or not isinstance(body, dict):
        return None
    items = body.get("observations") or []
    if not isinstance(items, list):
        return None
    if period_id:
        for item in items:
            if isinstance(item, dict) and _period(item) == period_id:
                return item
        return None
    rows = [item for item in items if isinstance(item, dict)]
    if len(rows) == 1 and _scalar_slot(_period(rows[0])):
        return rows[0]
    return None


def _scalar_slot(period: str) -> bool:
    """An empty period and the parser's params column key ``value`` are the same scalar."""
    return period == "" or period.casefold() == "value"


def _jobs(parser: str) -> dict[str, str]:
    status, body = _request_json("GET", f"{parser}/v1/context-jobs?status=succeeded", None, 30)
    found: dict[str, str] = {}
    if status != 200 or not isinstance(body, list):
        print(f"jobs -> {status}", file=sys.stderr)
        return found
    for item in body:
        if not isinstance(item, dict):
            continue
        name = str(item.get("source_filename") or "")
        job_id = str(item.get("job_id") or "")
        if name in _BOOKS and job_id and name not in found:
            found[name] = job_id
    return found


def _empty_result(case: dict[str, Any], note: str) -> dict[str, Any]:
    result = score_case(case, {}, lambda _row, _period: None)
    result["attribute_ok"] = False
    result["value_ok"] = False
    result["passed"] = False
    result["note"] = note
    return result


def _corpus_note(corpus: Path, document: dict[str, Any]) -> dict[str, Any]:
    books = sorted({str(case["book"]) for case in document["cases"]})
    skipped = [
        str(item.get("file")) for item in document.get("skipped") or [] if isinstance(item, dict)
    ]
    return {
        "dir": str(corpus),
        "books": {name: (corpus / name).is_file() for name in books},
        "skipped_present": {name: (corpus / name).is_file() for name in skipped},
    }


def _write_report(run_dir: Path, report: dict[str, Any]) -> None:
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(_markdown(report), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    cases = list(report["cases"])
    passed = sum(1 for item in cases if item["passed"])
    lines = [
        "# Бизнес-проверки качества ответов",
        "",
        f"Стенд готов: {'да' if report['ready'] else 'нет'}.",
        f"Агент: `{report['agent']}`. Парсер: `{report['parser']}`.",
        "",
        _book_line(report),
        _skip_line(report),
        "",
        f"Кейсов: {len(cases)}. Зелёных (атрибут и значение): {passed}. "
        f"Красных: {len(cases) - passed}.",
        "Формулировка в допуск не входит: дыра масштаба не отменяет верную ячейку.",
        "",
        "| Кейс | Атрибут | Значение | Формулировка | Итог |",
        "|---|---|---|---|---|",
    ]
    for item in cases:
        lines.append(
            f"| `{item['id']}` | {_yes(item['attribute_ok'])} | {_yes(item['value_ok'])} | "
            f"{_yes(item['wording_ok'])} | {_mark(item['passed'])} |"
        )
    lines.extend(["", *_phase_lines(cases), "", *_failure_lines(cases), ""])
    return "\n".join(lines)


def _book_line(report: dict[str, Any]) -> str:
    jobs = report.get("jobs") or {}
    corpus = report.get("corpus") or {}
    present = corpus.get("books") or {}
    parts = []
    for name in sorted(present):
        state = "файл корпуса есть" if present[name] else "файла корпуса нет"
        job = jobs.get(name) or "задача не найдена"
        parts.append(f"`{name}` ({state}, задача `{job}`)")
    return "Книги: " + "; ".join(parts) + "."


def _skip_line(report: dict[str, Any]) -> str:
    corpus = report.get("corpus") or {}
    present = corpus.get("skipped_present") or {}
    lines = []
    for item in report.get("skipped") or []:
        name = str(item.get("file") or "")
        found = "файл в корпусе есть" if present.get(name) else "файла в корпусе нет"
        lines.append(f"`{name}` не прогонялся: {item.get('reason')} {found}.")
    return " ".join(lines)


def _phase_lines(cases: list[dict[str, Any]]) -> list[str]:
    phase = [item for item in cases if item.get("group") == "phase"]
    lines = ["## Временная ось", ""]
    if not phase:
        lines.append("Кейсов про фазу нет.")
        return lines
    missed = [item for item in phase if not item["passed"]]
    if not missed:
        lines.append("Фразы про фазу попали в целевой год.")
    else:
        lines.append("Фраза про фазу не попала в бизнес-год. Агент в этом прогоне не менялся.")
        lines.append("")
        for item in missed:
            expect = item.get("expect") or {}
            target = f"{expect.get('row_key')} {expect.get('period_id')}"
            lines.append(
                f"- `{item['id']}`: цель `{target}`. {_cited(item)} {item.get('note') or ''}"
            )
    why = [item for item in cases if item.get("group") == "composition"]
    if why:
        lines.append("")
        lines.append("Состав числа (числа формулы и кэша входов входят в допуск):")
        for item in why:
            flag = "да" if item.get("precedent") else "нет"
            lines.append(f"- `{item['id']}`: предшественники или дыра precedent — {flag}.")
    return lines


def _failure_lines(cases: list[dict[str, Any]]) -> list[str]:
    failed = [item for item in cases if not item["passed"] or not item.get("wording_ok")]
    lines = ["## Расхождения", ""]
    if not failed:
        lines.append("Расхождений нет.")
        return lines
    for item in failed:
        excerpt = " ".join(str(item.get("content") or "").split())[:240]
        lines.append(
            f"- `{item['id']}` ({_mark(item['passed'])}, формулировка {_yes(item['wording_ok'])}): "
            f"{item.get('note') or 'без заметки'}"
        )
        if excerpt:
            lines.append(f"  Ответ: {excerpt}")
    return lines


def _cited(item: dict[str, Any]) -> str:
    citations = item.get("citations") or []
    if not citations:
        return "Цитаты нет."
    parts = [f"{_row(cite)} {_period(cite) or 'скаляр'}" for cite in citations]
    return "В ответе: " + ", ".join(parts) + "."


def _yes(flag: bool) -> str:
    return "да" if flag else "нет"


def _mark(flag: bool) -> str:
    return "зелёный" if flag else "красный"


def _progress(result: dict[str, Any]) -> None:
    print(
        f"{result['id']} http={result.get('http_status')} {result.get('seconds')}s "
        f"attribute={_yes(result['attribute_ok'])} value={_yes(result['value_ok'])} "
        f"wording={_yes(result['wording_ok'])}",
        file=sys.stderr,
        flush=True,
    )


def _base(name: str, fallback: str, default: str) -> str:
    raw = os.environ.get(name) or (os.environ.get(fallback) if fallback else "") or default
    return raw.rstrip("/")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())
    except TimeoutError:
        return 0, {"error": "timeout"}
    except urllib.error.URLError as exc:
        return 0, {"error": str(getattr(exc, "reason", exc))[:200]}


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {"error": "not_json"}


if __name__ == "__main__":
    sys.exit(main())
