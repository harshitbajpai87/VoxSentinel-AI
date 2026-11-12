"""Routes for the VoxSentinel Assistant (Gemini-backed in-app help).

Two endpoints only:

* ``GET  /api/v1/assistant/status`` — is the assistant configured, and what are
  its limits. Never exposes the key.
* ``POST /api/v1/assistant/chat``   — one conversational turn.

The response is always Gemini's own text. When Gemini cannot answer, the client
receives a friendly error and no assistant message, so the UI never shows an
invented answer.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.assistant.service import (
    AssistantError,
    AssistantUnavailable,
    get_assistant,
)

logger = logging.getLogger("voxsentinel.assistant.api")

router = APIRouter(prefix="/assistant", tags=["assistant"])


class ChatMessage(BaseModel):
    role: str = Field(default="user", max_length=16)
    content: str = Field(default="", max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=40)
    route: str | None = Field(default=None, max_length=200)
    page: str | None = Field(default=None, max_length=120)


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for") or ""
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


@router.get("/status")
def assistant_status() -> dict[str, Any]:
    service = get_assistant()
    described = service.describe()
    return {
        "available": bool(described.get("configured")),
        "provider": described.get("provider"),
        "provider_label": described.get("provider_label"),
        "model": described.get("model"),
        "enabled": described.get("enabled"),
        "limits": described.get("limits"),
    }


@router.post("/chat")
def assistant_chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    service = get_assistant()
    if not service.configured:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "assistant_not_configured",
                "message": (
                    "The VoxSentinel Assistant is not configured on this deployment. "
                    "An administrator needs to set GEMINI_API_KEY on the server."
                ),
            },
        )

    client_key = _client_key(request)
    try:
        service.check_rate(client_key)
    except AssistantError as exc:
        raise HTTPException(
            status_code=exc.http_status, detail={"code": exc.code, "message": exc.message}
        ) from exc

    messages = [{"role": item.role, "content": item.content} for item in payload.messages]
    try:
        result = service.chat(messages, path=payload.route or None)
    except AssistantError as exc:
        raise HTTPException(
            status_code=exc.http_status, detail={"code": exc.code, "message": exc.message}
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak internals to the browser
        logger.warning("assistant chat crashed: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "assistant_unavailable",
                "message": "The assistant is temporarily unavailable. Please try again.",
            },
        ) from exc

    service.record_hit(client_key)
    return result


__all__ = ["router", "AssistantUnavailable"]
