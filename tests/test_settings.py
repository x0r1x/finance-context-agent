"""Settings defaults, .env-style overrides, and range checks."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from finance_context_agent.settings import Settings, field_default

_KEYS = (
    "REDIS_URL",
    "PARSER_BASE_URL",
    "PARSER_TIMEOUT_SEC",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TIMEOUT_SEC",
    "LLM_TEMPERATURE",
    "HOST",
    "PORT",
    "CHECKPOINT_TTL_MINUTES",
    "CHECKPOINT_REFRESH_ON_READ",
    "LOCK_TTL_SEC",
    "RECURSION_LIMIT",
    "CONTENT_BUDGET",
    "CLARIFY_BUDGET",
    "OBSERVATION_CAP",
    "CATALOG_SEARCH_LIMIT",
    "PRECEDENT_DEPTH",
    "MAX_NEEDLES",
    "MAX_DEPENDENT_ROWS",
    "DEPENDENT_OBSERVATION_LIMIT",
    "LANGGRAPH_STRICT_MSGPACK",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_match_today(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.parser_base_url == "http://127.0.0.1:8080"
    assert settings.parser_timeout_sec == 15
    assert settings.llm_base_url == "http://127.0.0.1:1234/v1"
    assert settings.llm_api_key == ""
    assert settings.llm_model == "local"
    assert settings.llm_timeout_sec == 60
    assert settings.llm_temperature == 0
    assert settings.host == "0.0.0.0"
    assert settings.port == 8090
    assert settings.checkpoint_ttl_minutes == 1440
    assert settings.checkpoint_refresh_on_read is True
    assert settings.lock_ttl_sec == 900
    assert settings.recursion_limit == 40
    assert settings.content_budget == 4
    assert settings.clarify_budget == 2
    assert settings.observation_cap == 48
    assert settings.catalog_search_limit == 8
    assert settings.precedent_depth == 2
    assert settings.max_needles == 4
    assert settings.max_dependent_rows == 4
    assert settings.dependent_observation_limit == 8
    assert settings.langgraph_strict_msgpack is True
    assert settings.saver_ttl() == {"default_ttl": 1440, "refresh_on_read": True}
    assert os.environ["LANGGRAPH_STRICT_MSGPACK"] == "true"
    assert field_default("parser_timeout_sec") == 15
    assert field_default("observation_cap") == 48


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OBSERVATION_CAP", "24")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("CHECKPOINT_REFRESH_ON_READ", "false")
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "false")
    settings = Settings(_env_file=None)
    assert settings.observation_cap == 24
    assert settings.llm_temperature == 0.2
    assert settings.checkpoint_refresh_on_read is False
    assert settings.langgraph_strict_msgpack is False
    assert settings.saver_ttl()["refresh_on_read"] is False
    assert os.environ["LANGGRAPH_STRICT_MSGPACK"] == "false"


def test_blank_value_keeps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("CONTENT_BUDGET", "  ")
    monkeypatch.setenv("PARSER_BASE_URL", "")
    settings = Settings(_env_file=None)
    assert settings.content_budget == 4
    assert settings.parser_base_url == "http://127.0.0.1:8080"


def test_out_of_range_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OBSERVATION_CAP", "49")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
