"""
Error handlers for the API.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from src.api.schemas import ErrorResponse
import logging
from typing import Union

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=str(exc.detail),
            details={"status_code": exc.status_code, "path": str(request.url.path)}
        ).model_dump()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    errors = exc.errors()
    error_messages = [f"{err['loc']}: {err['msg']}" for err in errors]
    logger.warning(f"Validation error: {error_messages} - Path: {request.url.path}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            message="Invalid request data",
            details={"errors": errors, "path": str(request.url.path)}
        ).model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    error_type = type(exc).__name__
    error_message = str(exc)
    logger.exception(
        f"Unhandled exception: {error_type} - {error_message} - Path: {request.url.path}",
        exc_info=True
    )
    
    # Don't expose internal error details in production
    message = "An unexpected error occurred. Please try again later."
    details = {"type": error_type}
    
    # In development, include more details
    import os
    if os.getenv("ENV", "production") == "development":
        details["message"] = error_message
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=message,
            details=details
        ).model_dump()
    )

