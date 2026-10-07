"""Settings defaults, .env-style overrides, and range checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from finance_context_agent.settings import Settings, field_default, normalize_base_url

_SRC = Path(__file__).resolve().parents[1] / "src"
_MSGPACK_PROBE = (
    "import finance_context_agent.app\n"
    "from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED\n"
    "print(int(STRICT_MSGPACK_ENABLED))\n"
)

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
    assert settings.catalog_search_limit == 32
    assert settings.precedent_depth == 2
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


def _probe_msgpack(cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("LANGGRAPH_STRICT_MSGPACK", None)
    env["PYTHONPATH"] = os.pathsep.join((str(_SRC), env.get("PYTHONPATH", ""))).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", _MSGPACK_PROBE],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_msgpack_flag_is_on_when_langgraph_imports(tmp_path: Path) -> None:
    result = _probe_msgpack(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_dotenv_msgpack_flag_reaches_langgraph(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LANGGRAPH_STRICT_MSGPACK=false\n")
    result = _probe_msgpack(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


class _DockerPath:
    def __init__(self, value: object) -> None:
        self._value = str(value)

    def exists(self) -> bool:
        return self._value == "/.dockerenv"


class _HostPath:
    def __init__(self, value: object) -> None:
        self._value = str(value)

    def exists(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("kind", "env", "docker", "raw", "expected"),
    [
        pytest.param(
            "parser",
            None,
            True,
            None,
            "http://host.docker.internal:8080",
            id="loopback-inside-docker",
        ),
        pytest.param(
            "parser",
            "http://user:secret@localhost:9090/custom?x=1",
            True,
            None,
            "http://user:secret@host.docker.internal:9090/custom?x=1",
            id="keeps-port-path-query-userinfo",
        ),
        pytest.param(
            "parser", None, False, None, "http://127.0.0.1:8080", id="loopback-outside-docker"
        ),
        pytest.param(
            "parser",
            "http://parser:8080/root",
            True,
            None,
            "http://parser:8080/root",
            id="non-loopback-inside-docker",
        ),
        pytest.param(
            "model",
            None,
            False,
            "http://127.0.0.1:1234",
            "http://127.0.0.1:1234/v1",
            id="empty-model-base-gains-v1",
        ),
    ],
)
def test_base_url(
    kind: str,
    env: str | None,
    docker: bool,
    raw: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if kind == "model":
        assert normalize_base_url(raw or "") == expected
        return
    _clear(monkeypatch)
    if env is not None:
        monkeypatch.setenv("PARSER_BASE_URL", env)
    monkeypatch.setattr(
        "finance_context_agent.settings.Path", _DockerPath if docker else _HostPath
    )
    assert Settings(_env_file=None).resolved_parser_base_url() == expected
