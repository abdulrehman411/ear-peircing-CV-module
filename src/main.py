"""
FastAPI application entry point.
"""
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from src.config import get_settings
from src.api.routes import router
from src.api.error_handlers import http_exception_handler, general_exception_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Ear Piercing CV Module",
    description="Computer Vision module for ear piercing alignment and validation",
    version="1.0.0"
)

# Configure CORS
origins = settings.cors_origins.split(",") if settings.cors_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API routes
app.include_router(router, prefix=f"/api/{settings.api_version}", tags=["CV"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Ear Piercing CV Module API",
        "version": "1.0.0",
        "docs": "/docs"
    }

