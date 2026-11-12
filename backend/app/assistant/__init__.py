"""VoxSentinel Assistant: server-side Gemini chat for the in-app helper.

The browser never sees the API key. The frontend posts a short transcript plus
the page it is on; this module adds the system prompt and the page context,
calls Gemini, and returns only the model's own text. When Gemini cannot answer,
the caller gets a typed error and the UI shows a friendly message — no answer
is ever invented here.
"""

from app.assistant.service import (
    AssistantConfig,
    AssistantError,
    AssistantUnavailable,
    AssistantService,
    RateLimited,
    get_assistant,
)

__all__ = [
    "AssistantConfig",
    "AssistantError",
    "AssistantUnavailable",
    "AssistantService",
    "RateLimited",
    "get_assistant",
]
