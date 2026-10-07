"""HTTP handlers. Each module owns its paths; this package adds no prefix."""

from fastapi import APIRouter

from finance_context_agent.api.routes.chat import router as chat_router
from finance_context_agent.api.routes.health import router as health_router
from finance_context_agent.api.routes.models import router as models_router

router = APIRouter()
router.include_router(health_router)
router.include_router(models_router)
router.include_router(chat_router)
