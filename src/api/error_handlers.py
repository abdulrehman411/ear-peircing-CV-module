"""
Error handlers for the API.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from src.api.schemas import ErrorResponse
import logging

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__}
        ).model_dump()
    )

