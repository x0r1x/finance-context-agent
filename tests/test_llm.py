import json

import httpx
import pytest

from finance_context_agent.llm import ModelError, OpenAIChat, SchemaError, chat_completions_url


def test_chat_completions_url_appends_the_missing_suffix() -> None:
    assert chat_completions_url("http://localhost:1234") == (
        "http://localhost:1234/v1/chat/completions"
    )
    assert chat_completions_url("http://localhost:1234/v1") == (
        "http://localhost:1234/v1/chat/completions"
    )
    assert chat_completions_url("http://localhost:1234/v1/chat/completions") == (
        "http://localhost:1234/v1/chat/completions"
    )


def _chat(handler) -> OpenAIChat:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChat("http://llm/v1", "key", "local", client=client)


@pytest.mark.asyncio
async def test_invalid_json_is_retried_once() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            body = {"choices": [{"message": {"content": "not-json"}}]}
        else:
            body = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        return httpx.Response(200, json=body)

    chat = _chat(handler)
    parsed = await chat.complete_json(role="plan", system="s", user="u")
    assert parsed == {"ok": True}
    assert calls["n"] == 2
    await chat.aclose()


@pytest.mark.asyncio
async def test_two_invalid_payloads_raise_schema_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    chat = _chat(handler)
    with pytest.raises(SchemaError):
        await chat.complete_json(role="plan", system="s", user="u")
    assert calls["n"] == 2
    await chat.aclose()


@pytest.mark.asyncio
async def test_chat_completions_body_uses_json_schema() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        role = payload["response_format"]["json_schema"]["name"]
        if role == "plan":
            content = (
                '{"question_type":"lookup","needles":["EBITDA"],"periods":[],"trace":"none"}'
            )
        elif role == "answer":
            content = '{"text":"0","citations":[]}'
        else:
            content = '{"gaps":[]}'
        body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return httpx.Response(200, json=body)

    chat = _chat(handler)
    await chat.complete_json(role="plan", system="s", user="u")
    await chat.complete_json(role="answer", system="s", user="u")
    await chat.complete_json(role="critic", system="s", user="u")
    await chat.aclose()

    assert [item["response_format"]["json_schema"]["name"] for item in seen] == [
        "plan",
        "answer",
        "critic",
    ]
    for payload in seen:
        assert list(payload) == ["model", "temperature", "messages", "response_format"]
        assert payload["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        assert all(isinstance(item["content"], str) for item in payload["messages"])
        assert "input" not in payload and "text" not in payload
        saved = payload["response_format"]
        assert saved["type"] == "json_schema"
        assert saved["json_schema"]["strict"] is True
        assert "json_object" not in json.dumps(payload)
    plan_schema = seen[0]["response_format"]["json_schema"]["schema"]
    assert plan_schema["properties"]["question_type"]["enum"] == [
        "lookup",
        "compare",
        "explain",
        "compose",
    ]
    assert "flag" not in plan_schema["properties"]["periods"]["items"]["properties"]
    assert set(seen[1]["response_format"]["json_schema"]["schema"]["properties"]) == {
        "text",
        "citations",
    }
    assert set(seen[2]["response_format"]["json_schema"]["schema"]["properties"]) == {"gaps"}


@pytest.mark.asyncio
async def test_http_400_keeps_status_and_body_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            400,
            json={"error": "response_format.type must be json_schema or text"},
        )

    chat = _chat(handler)
    with pytest.raises(ModelError) as caught:
        await chat.complete_json(role="plan", system="s", user="u")
    await chat.aclose()
    assert calls["n"] == 1
    assert caught.value.status == 400
    assert "json_schema" in caught.value.body
    assert str(caught.value) == "400"


@pytest.mark.asyncio
async def test_http_error_is_not_retried_as_schema() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "down"})

    chat = _chat(handler)
    with pytest.raises(ModelError):
        await chat.complete_json(role="plan", system="s", user="u")
    assert calls["n"] == 1
    await chat.aclose()
