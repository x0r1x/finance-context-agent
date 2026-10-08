import httpx
import pytest

from finance_context_agent.clients.parser import ParserClient, ParserError


class _Script:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
        etag = '"book"'
        if request.method == "HEAD":
            return httpx.Response(200, headers={"etag": etag})
        if request.headers.get("if-none-match") == etag:
            return httpx.Response(304, headers={"etag": etag})
        if url.endswith("/catalog?limit=1"):
            body = {"axes": [{"id": "forecast"}], "rows": [{"row_key": "only"}], "total": 4}
        elif "q=DSCR" in url:
            body = {
                "axes": [{"id": "forecast"}],
                "rows": [{"row_key": "dscr"}],
                "total": 1,
                "marker": "search",
            }
        elif url.endswith(
            "/observations?row_key=rk&precedent_depth=2&limit=24&period_id=2030&period_id=2031"
        ):
            body = {"observations": []}
        elif "/graph/trace?" in url:
            body = {"nodes": []}
        elif url.endswith("/summary"):
            body = {"coverage": {}}
        elif url.endswith("/context.json"):
            body = {"meta": {"source_filename": "model.xlsx"}, "blocks": []}
        elif url.endswith("/context-jobs?status=succeeded"):
            body = [{"job_id": "job-1", "source_filename": "model.xlsx"}]
        elif url.endswith("/readyz"):
            return httpx.Response(200, json={"status": "ready"})
        else:
            return httpx.Response(500, json={"error": "unexpected", "url": url})
        return httpx.Response(200, json=body, headers={"etag": etag})


def _client(script: _Script) -> ParserClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(script.handler))
    return ParserClient("http://parser", client=http)


@pytest.mark.asyncio
async def test_if_none_match_is_bound_to_the_full_url() -> None:
    script = _Script()
    parser = _client(script)
    axes, etag = await parser.get_axes("job-1")
    assert etag == '"book"'
    assert axes["rows"][0]["row_key"] == "only"

    found = await parser.search_rows("job-1", "DSCR")
    search_call = script.calls[-1]
    assert "if-none-match" not in {key.lower() for key in search_call.headers}
    assert found["marker"] == "search"

    again, _etag = await parser.get_axes("job-1")
    assert again["rows"][0]["row_key"] == "only"
    assert script.calls[-1].headers["if-none-match"] == '"book"'

    searched = await parser.search_rows("job-1", "DSCR")
    assert searched["marker"] == "search"
    assert script.calls[-1].headers["if-none-match"] == '"book"'
    await parser.aclose()


@pytest.mark.asyncio
async def test_observations_use_one_selector_and_repeat_periods() -> None:
    script = _Script()
    parser = _client(script)
    await parser.get_observations(
        "job-1",
        row_key="rk",
        period_ids=["2030", "2031"],
        precedent_depth=2,
        limit=24,
    )
    url = str(script.calls[-1].url)
    assert "row_key=rk" in url
    assert "period_id=2030" in url
    assert "period_id=2031" in url
    assert "concept_id" not in url
    assert "q=" not in url
    await parser.aclose()


@pytest.mark.asyncio
async def test_trace_query_uses_from() -> None:
    script = _Script()
    parser = _client(script)
    await parser.trace_dependents("job-1", "rk", depth=2)
    url = str(script.calls[-1].url)
    assert "from=rk" in url
    assert "direction=dependents" in url
    assert "depth=2" in url
    await parser.aclose()


@pytest.mark.asyncio
async def test_list_rows_joins_pages_without_concept_or_query() -> None:
    pages = {
        0: [
            {
                "row_key": str(index),
                "label": "CAPEX" if index == 0 else f"Row {index}",
                "concept_id": "secret",
                "sheet": "Input",
                "label_path": ["COSTS"],
                "axis_ids": ["a"],
                "disposition": "fact",
                "kind": "fact",
            }
            for index in range(1000)
        ],
        1000: [
            {
                "row_key": "tail",
                "label": "DSCR",
                "concept_id": "cov.dscr",
                "sheet": "Ratios",
                "label_path": [],
                "axis_ids": [],
                "disposition": "fact",
                "kind": "fact",
            },
            {"row_key": "blank", "label": "  ", "concept_id": "hidden"},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        assert "q=" not in url
        offset = int(request.url.params["offset"])
        body = pages.get(offset)
        if body is None:
            return httpx.Response(500, json={"error": "unexpected", "url": url})
        return httpx.Response(200, json={"total": 1002, "rows": body})

    parser = ParserClient(
        "http://parser", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    rows = await parser.list_rows("job-1")
    assert rows[0]["label"] == "CAPEX"
    assert rows[-1]["row_key"] == "tail"
    assert all("concept_id" not in row for row in rows)
    assert len(rows) == 1001
    await parser.aclose()


@pytest.mark.asyncio
async def test_error_body_becomes_parser_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "selector_required"})

    parser = ParserClient(
        "http://parser", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ParserError) as caught:
        await parser.get_observations("job-1", row_key="")
    assert caught.value.status == 400
    assert caught.value.code == "selector_required"
    await parser.aclose()


@pytest.mark.asyncio
async def test_jobs_must_be_a_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jobs": []})

    parser = ParserClient(
        "http://parser", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ParserError) as caught:
        await parser.list_jobs(status="succeeded")
    assert caught.value.code == "bad_jobs"
    await parser.aclose()


@pytest.mark.asyncio
async def test_head_does_not_send_a_cached_etag() -> None:
    script = _Script()
    parser = _client(script)
    await parser.get_axes("job-1")
    etag = await parser.head_catalog_etag("job-1")
    assert etag == '"book"'
    assert "if-none-match" not in {key.lower() for key in script.calls[-1].headers}
    document = await parser.get_context("job-1")
    assert document["meta"]["source_filename"] == "model.xlsx"
    assert str(script.calls[-1].url).endswith("/v1/context-jobs/job-1/context.json")
    assert "if-none-match" not in {key.lower() for key in script.calls[-1].headers}
    await parser.get_context("job-1")
    assert "if-none-match" not in {key.lower() for key in script.calls[-1].headers}
    assert not parser.remembers("/v1/context-jobs/job-1/context.json", [])
    await parser.aclose()
