"""Question agent over finance-context-builder slices."""

from finance_context_agent.settings import Settings

__version__ = "0.1.0"

# LangGraph copies LANGGRAPH_STRICT_MSGPACK when its serde module is first imported.
# This package module loads before that import, so the resolved value is already set.
Settings()
