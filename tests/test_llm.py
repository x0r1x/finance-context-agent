import json

import httpx
import pytest

from finance_context_agent.clients.llm import (
    _MODELS,
    ModelError,
    OpenAIChat,
    Plan,
    SchemaError,
    chat_completions_url,
    response_format_for,
)


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
            body = {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"question_type":"lookup","needles":["EBITDA"],'
                                '"periods":[],"trace":"none"}'
                            )
                        }
                    }
                ]
            }
        return httpx.Response(200, json=body)

    chat = _chat(handler)
    parsed = await chat.complete_json(role="plan", system="s", user="u")
    assert parsed == {
        "question_type": "lookup",
        "needles": ["EBITDA"],
        "periods": [],
        "trace": "none",
    }
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
        elif role == "choose":
            content = '{"option":"1"}'
        elif role == "about":
            content = '{"acts":["chitchat"]}'
        elif role == "talk":
            content = '{"act":"catalog"}'
        else:
            content = '{"gaps":[]}'
        body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return httpx.Response(200, json=body)

    chat = _chat(handler)
    await chat.complete_json(role="plan", system="s", user="u")
    await chat.complete_json(role="answer", system="s", user="u")
    await chat.complete_json(role="choose", system="s", user="u", options=["1", "2"])
    await chat.complete_json(role="about", system="s", user="u")
    await chat.complete_json(role="talk", system="s", user="u")
    await chat.aclose()

    assert [item["response_format"]["json_schema"]["name"] for item in seen] == [
        "plan",
        "answer",
        "choose",
        "about",
        "talk",
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
    period = plan_schema["properties"]["periods"]["items"]
    assert "flag" not in period["properties"]
    assert period["properties"]["phase_year"]["type"] == "string"
    assert "книге" in plan_schema["properties"]["needles"]["description"]
    for payload in seen:
        _assert_strict_schema(payload["response_format"]["json_schema"]["schema"])
    assert set(seen[1]["response_format"]["json_schema"]["schema"]["properties"]) == {
        "text",
        "citations",
    }
    choose = seen[2]["response_format"]["json_schema"]["schema"]
    assert choose["properties"]["option"]["enum"] == ["none", "1", "2"]
    about = seen[3]["response_format"]["json_schema"]["schema"]
    talk = seen[4]["response_format"]["json_schema"]["schema"]
    assert set(talk["properties"]) == {"act"}
    assert "sheet" not in talk["properties"]
    assert "label" not in talk["properties"]
    assert talk["properties"]["act"]["enum"] == [
        "files",
        "overview",
        "catalog",
        "figure",
        "greet",
        "explain",
        "unclear",
    ]
    assert set(about["properties"]) == {"acts"}
    assert about["properties"]["acts"]["items"]["enum"] == ["chitchat", "books", "book", "row"]
    assert about["additionalProperties"] is False
    assert about["required"] == ["acts"]
    refused = {"n": 0}

    def reject(request: httpx.Request) -> httpx.Response:
        refused["n"] += 1
        body = {"choices": [{"message": {"content": '{"option":"9"}'}}]}
        return httpx.Response(200, json=body)

    closed = _chat(reject)
    with pytest.raises(SchemaError):
        await closed.complete_json(role="choose", system="s", user="u", options=["1", "2"])
    assert refused["n"] == 2
    await closed.aclose()


def _assert_strict_schema(node: object) -> None:
    if isinstance(node, dict):
        assert "$ref" not in node
        assert "$defs" not in node
        for item in node.get("anyOf") or []:
            if isinstance(item, dict):
                assert item.get("type") != "null"
        if node.get("type") == "object":
            properties = node["properties"]
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(properties)
        for value in node.values():
            _assert_strict_schema(value)
    elif isinstance(node, list):
        for item in node:
            _assert_strict_schema(item)


@pytest.mark.asyncio
async def test_view_schema_is_the_orientations_of_this_frame() -> None:
    assert "view" not in _MODELS
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = {"choices": [{"message": {"content": '{"view":"table-period"}'}}]}
        return httpx.Response(200, json=body)

    chat = _chat(handler)
    parsed = await chat.complete_json(
        role="view",
        system="s",
        user="u",
        options=["sentence", "table-period"],
    )
    await chat.aclose()
    assert parsed == {"view": "table-period"}
    schema = seen[0]["response_format"]["json_schema"]["schema"]
    assert schema == response_format_for("view", ["sentence", "table-period"])["json_schema"][
        "schema"
    ]
    assert schema["properties"]["view"]["enum"] == ["sentence", "table-period"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["view"]
    calls = {"n": 0}

    def reject(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = {"choices": [{"message": {"content": '{"view":"table-label","index":0}'}}]}
        return httpx.Response(200, json=body)

    closed = _chat(reject)
    with pytest.raises(SchemaError):
        await closed.complete_json(
            role="view",
            system="s",
            user="u",
            options=["sentence", "table-period"],
        )
    assert calls["n"] == 2
    await closed.aclose()


def test_blank_period_is_absent_and_a_partial_one_stays() -> None:
    blank = Plan.model_validate(
        {
            "question_type": "lookup",
            "needles": ["EBITDA"],
            "periods": [{"period_key": "", "year": "  ", "phase_year": ""}],
            "trace": "none",
        }
    )
    assert blank.model_dump(mode="json")["periods"] == []
    named = Plan.model_validate(
        {
            "question_type": "lookup",
            "needles": ["EBITDA"],
            "periods": [{"year": "Y1", "phase_year": "1"}],
            "trace": "none",
        }
    )
    assert named.model_dump(mode="json")["periods"] == [
        {"period_key": "", "year": "Y1", "phase_year": "1"}
    ]
    punctuation = Plan.model_validate(
        {
            "question_type": "lookup",
            "needles": ["EBITDA"],
            "periods": [{"period_key": "Y1", "year": ", ", "phase_year": ","}],
            "trace": "none",
        }
    )
    assert punctuation.model_dump(mode="json")["periods"] == [
        {"period_key": "Y1", "year": "", "phase_year": ""}
    ]


@pytest.mark.asyncio
async def test_answer_without_citations_raises_schema_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"text": "0"}'}}]})

    chat = _chat(handler)
    with pytest.raises(SchemaError):
        await chat.complete_json(role="answer", system="s", user="u")
    assert calls["n"] == 2
    await chat.aclose()


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
