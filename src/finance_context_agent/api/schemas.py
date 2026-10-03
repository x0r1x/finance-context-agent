"""Public chat body. Unknown fields are ignored."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = []
    stream: bool = False
    job_id: str | None = None
    thread_id: str | None = None


def last_user_text(messages: list[ChatMessage]) -> str:
    if not messages:
        raise ValueError("messages_required")
    last = messages[-1]
    if last.role != "user":
        raise ValueError("last_message_not_user")
    text = _content_text(last.content).strip()
    if not text:
        raise ValueError("user_message_required")
    return text


def _content_text(content: Any) -> str:
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
