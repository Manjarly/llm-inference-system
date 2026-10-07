"""FastAPI application factory for the LLM inference server."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from inference.config import EngineConfig
from inference.engine.engine import LLMInferenceEngine
from inference.server.api import router as api_router


def create_app(
    engine: Optional[LLMInferenceEngine] = None,
    config: Optional[EngineConfig] = None,
) -> FastAPI:
    """Create and configure the FastAPI server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        nonlocal engine
        if engine is None:
            cfg = config or EngineConfig()
            engine = LLMInferenceEngine(cfg)
        app.state.engine = engine
        engine.start()
        yield
        # Shutdown
        if engine:
            engine.stop()

    app = FastAPI(
        title="High-Performance LLM Inference Server",
        description="Production LLM Serving with Continuous Batching, Quantization, GPU Telemetry & Load Testing",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for web apps & dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(api_router)

    # Static files for dashboard
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/dashboard")

    return app
