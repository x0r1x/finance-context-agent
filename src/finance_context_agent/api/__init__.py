"""HTTP routes. Paths stay on the handlers. This package adds no prefix."""

from fastapi import APIRouter

from finance_context_agent.api.chat import router as chat_router
from finance_context_agent.api.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(chat_router)
