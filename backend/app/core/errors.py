"""Consistent error responses.

Contract shape (docs/API_CONTRACT.md section 4):

    {"error": {"code": "file_too_large", "message": "Safe, user-facing text"}}

Internal details (tracebacks, paths, secrets) are logged server-side only and
never returned to clients.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("voxsentinel.errors")

STATUS_TO_CODE = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "file_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "model_unavailable",
}

SAFE_DEFAULT_MESSAGES = {
    400: "The request was malformed.",
    404: "The requested resource was not found.",
    413: "The uploaded file is too large.",
    415: "The uploaded file type is not supported.",
    422: "The request could not be validated.",
    500: "An unexpected server error occurred.",
    503: "The detection model is not available.",
}


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            code = str(detail.get("code", "error"))
            message = str(
                detail.get("message", SAFE_DEFAULT_MESSAGES.get(exc.status_code, "Request failed."))
            )
        else:
            code = STATUS_TO_CODE.get(exc.status_code, "error")
            message = (
                str(detail)
                if isinstance(detail, str) and detail
                else SAFE_DEFAULT_MESSAGES.get(exc.status_code, "Request failed.")
            )
        return JSONResponse(status_code=exc.status_code, content=error_body(code, message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field NAMES and failure kinds are safe and actionable ("test: field
        # required"); the user's INPUT VALUES are never echoed back. The previous
        # blanket "one or more request fields are invalid" left the UI unable to
        # say what was missing.
        errors = exc.errors()
        logger.info("Request validation failed: %s", errors)
        fields: list[str] = []
        for err in errors:
            location = [
                str(part)
                for part in (err.get("loc") or ())
                if part not in ("body", "query", "path", "header")
            ]
            name = ".".join(location) or "request"
            reason = {
                "missing": "is required",
                "value_error": "has an invalid value",
                "type_error": "has the wrong type",
                "bool_parsing": "must be true or false",
                "int_parsing": "must be a whole number",
                "float_parsing": "must be a number",
                "file_required": "is required",
            }.get(str(err.get("type")), "is invalid")
            entry = f"{name} {reason}"
            if entry not in fields:
                fields.append(entry)
        if fields:
            message = "Check the request fields: " + "; ".join(fields[:5]) + "."
        else:
            message = "One or more request fields are invalid."
        return JSONResponse(
            status_code=422,
            content=error_body("validation_error", message),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Server-side only: the traceback never leaves the process.
        logger.exception("Unhandled server error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=error_body("internal_error", SAFE_DEFAULT_MESSAGES[500]),
        )
