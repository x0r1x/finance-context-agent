"""Model list for an OpenAI-compatible chat client."""

from __future__ import annotations

from fastapi import APIRouter

from finance_context_agent.session import AGENT_MODEL

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": AGENT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": AGENT_MODEL,
            }
        ],
    }
