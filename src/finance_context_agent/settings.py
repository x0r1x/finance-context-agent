"""Process settings. Postgres and SQLite are not configured."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


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
            userinfo = parts.username
            if parts.password is not None:
                userinfo += f":{parts.password}"
            userinfo += "@"
        hostport = rewrite_loopback_to
        if parts.port is not None:
            hostport = f"{hostport}:{parts.port}"
        netloc = f"{userinfo}{hostport}"
    return urlunsplit((parts.scheme, netloc, path, parts.query, parts.fragment))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    parser_base_url: str = "http://127.0.0.1:8080"
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = ""
    llm_model: str = "local"
    host: str = "0.0.0.0"
    port: int = 8090

    def resolved_llm_base_url(self) -> str:
        rewrite = "host.docker.internal" if Path("/.dockerenv").exists() else None
        return normalize_base_url(self.llm_base_url, rewrite_loopback_to=rewrite)
