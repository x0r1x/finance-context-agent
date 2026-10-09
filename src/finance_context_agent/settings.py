"""Process settings. Postgres and SQLite are not configured."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_BLANK_INTS = (
    "port",
    "checkpoint_ttl_minutes",
    "lock_ttl_sec",
    "recursion_limit",
    "content_budget",
    "clarify_budget",
    "observation_cap",
    "catalog_search_limit",
    "precedent_depth",
    "max_dependent_rows",
    "dependent_observation_limit",
)
_BLANK_FLOATS = ("parser_timeout_sec", "llm_timeout_sec", "llm_temperature")
_BLANK_BOOLS = ("checkpoint_refresh_on_read", "langgraph_strict_msgpack")
_BLANK_STRINGS = (
    "redis_url",
    "parser_base_url",
    "llm_base_url",
    "llm_model",
    "host",
    "ranker_base_url",
    "ranker_api_key",
    "ranker_model",
    "embed_model",
)


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def normalize_base_url(url: str, *, rewrite_loopback_to: str | None = None) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    if path == "":
        path = "/v1"
    netloc = parts.netloc
    host = parts.hostname
    if rewrite_loopback_to and host in _LOOPBACK:
        userinfo = ""
        if parts.username:
            userinfo += parts.username
            if parts.password is not None:
                userinfo += f":{parts.password}"
            userinfo += "@"
        hostport = rewrite_loopback_to
        if parts.port is not None:
            hostport = f"{hostport}:{parts.port}"
        netloc = f"{userinfo}{hostport}"
    return urlunsplit((parts.scheme, netloc, path, parts.query, parts.fragment))


def rewrite_loopback_host(url: str, *, rewrite_loopback_to: str | None = None) -> str:
    """Replace a loopback host and leave the path alone.

    Parser routes are joined onto the base, including ``/readyz``. An empty
    path must stay empty. ``normalize_base_url`` is for the model base, where
    an empty path means ``/v1``.
    """
    stripped = url.strip()
    if not rewrite_loopback_to:
        return stripped
    parts = urlsplit(stripped)
    if parts.hostname not in _LOOPBACK:
        return stripped
    userinfo = ""
    if parts.username:
        userinfo += parts.username
        if parts.password is not None:
            userinfo += f":{parts.password}"
        userinfo += "@"
    hostport = rewrite_loopback_to
    if parts.port is not None:
        hostport = f"{hostport}:{parts.port}"
    netloc = f"{userinfo}{hostport}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    parser_base_url: str = "http://127.0.0.1:8080"
    parser_timeout_sec: float = Field(default=15, gt=0)
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = ""
    llm_model: str = "local"
    llm_timeout_sec: float = Field(default=60, gt=0)
    llm_temperature: float = Field(default=0, ge=0, le=2)
    host: str = "0.0.0.0"
    port: int = 8090
    checkpoint_ttl_minutes: int = Field(default=1440, ge=1)
    checkpoint_refresh_on_read: bool = True
    lock_ttl_sec: int = Field(default=900, ge=1)
    recursion_limit: int = Field(default=40, ge=1)
    content_budget: int = Field(default=4, ge=1)
    clarify_budget: int = Field(default=2, ge=1)
    observation_cap: int = Field(default=48, ge=1, le=48)
    catalog_search_limit: int = Field(default=32, ge=1, le=100)
    precedent_depth: int = Field(default=2, ge=0, le=3)
    max_dependent_rows: int = Field(default=4, ge=1, le=8)
    dependent_observation_limit: int = Field(default=8, ge=1, le=48)
    langgraph_strict_msgpack: bool = True
    ranker_base_url: str = "http://127.0.0.1:8700/v1/systemone"
    ranker_api_key: str = ""
    ranker_model: str = "clm-latest"
    embed_model: str = "text-embedding-qwen3-embedding-4b"

    @field_validator(*_BLANK_STRINGS, mode="before")
    @classmethod
    def blank_string_keeps_default(cls, value: object, info: ValidationInfo) -> object:
        if _blank(value):
            return cls.model_fields[info.field_name].default
        return value

    @field_validator(*_BLANK_INTS, *_BLANK_FLOATS, *_BLANK_BOOLS, mode="before")
    @classmethod
    def blank_value_keeps_default(cls, value: object, info: ValidationInfo) -> object:
        if _blank(value):
            return cls.model_fields[info.field_name].default
        return value

    @model_validator(mode="after")
    def publish_msgpack(self) -> Settings:
        os.environ["LANGGRAPH_STRICT_MSGPACK"] = (
            "true" if self.langgraph_strict_msgpack else "false"
        )
        return self

    def resolved_llm_base_url(self) -> str:
        rewrite = "host.docker.internal" if Path("/.dockerenv").exists() else None
        return normalize_base_url(self.llm_base_url, rewrite_loopback_to=rewrite)

    def resolved_parser_base_url(self) -> str:
        rewrite = "host.docker.internal" if Path("/.dockerenv").exists() else None
        return rewrite_loopback_host(self.parser_base_url, rewrite_loopback_to=rewrite)

    def resolved_ranker_base_url(self) -> str:
        rewrite = "host.docker.internal" if Path("/.dockerenv").exists() else None
        return rewrite_loopback_host(self.ranker_base_url, rewrite_loopback_to=rewrite)

    def saver_ttl(self) -> dict[str, Any]:
        return {
            "default_ttl": self.checkpoint_ttl_minutes,
            "refresh_on_read": self.checkpoint_refresh_on_read,
        }


def field_default(name: str) -> Any:
    return Settings.model_fields[name].default
