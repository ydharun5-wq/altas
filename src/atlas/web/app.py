"""FastAPI application factory for private, local ATLAS installations."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from atlas.web import models as _models  # register mapped tables before migration/test setup
from atlas.web.api import router as api_router
from atlas.web.auth import router as auth_router
from atlas.web.config import Settings, get_settings
from atlas.web.database import make_engine, make_session_factory

_logger = logging.getLogger("atlas.web")


def create_app(*, settings: Settings | None = None, database_url: str | None = None) -> FastAPI:
    settings = settings or get_settings()
    if database_url is not None:
        settings = settings.model_copy(update={"database_url": database_url})
    settings.validate_runtime_security()
    engine = make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        engine.dispose()

    app = FastAPI(
        title="ATLAS V1",
        version="0.2.0",
        docs_url="/docs" if settings.enable_api_docs and settings.environment != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_api_docs and settings.environment != "production" else None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
    )

    @app.middleware("http")
    async def security_and_access_log(request: Request, call_next):
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _logger.exception("request failed method=%s path=%s", request.method, request.url.path)
            raise
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith(("/auth", "/api")) else "no-cache"
        _logger.info(json.dumps({
            "event": "http_request", "method": request.method,
            "path": request.url.path, "status": response.status_code,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        }, separators=(",", ":")))
        return response

    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, str]:
        return {"status": "alive", "mode": "paper_only"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> dict[str, str]:
        try:
            with request.app.state.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Database is not ready") from exc
        return {"status": "ready", "mode": "paper_only"}

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": "ATLAS V1", "mode": "private paper trading", "ui": "/app/"}

    app.include_router(auth_router)
    app.include_router(api_router)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        _logger.exception("unhandled application error method=%s path=%s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app
