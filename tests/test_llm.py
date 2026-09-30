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
