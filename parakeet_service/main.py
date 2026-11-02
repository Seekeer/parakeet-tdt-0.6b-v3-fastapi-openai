"""FastAPI application factory and resource lifespan."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .auth import ApiKeyMiddleware
from .batchworker import build_worker
from .config import API_KEY_DIGESTS, AUDIO_WORKERS, AUTH_DISABLED, DEFAULT_MODEL, logger
from .model import get_model, load_model
from .routes import public_router, router


def _shutdown_pool(pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True, cancel_futures=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.worker = None
    app.state.audio_pool = ThreadPoolExecutor(
        max_workers=AUDIO_WORKERS, thread_name_prefix="audio"
    )
    if AUTH_DISABLED:
        logger.warning("API key authentication is DISABLED (PARAKEET_AUTH_DISABLED=true)")
    else:
        logger.info("API key authentication enabled (%d key(s))", len(API_KEY_DIGESTS))
    try:
        logger.info("Lifespan startup: loading default model")
        await asyncio.to_thread(load_model, DEFAULT_MODEL)
        app.state.worker = build_worker(get_model)
        await app.state.worker.start()
        app.state.ready = True
        logger.info("Service ready")
        yield
    finally:
        app.state.ready = False
        logger.info("Lifespan shutdown")
        if app.state.worker is not None:
            await app.state.worker.stop()
        await asyncio.to_thread(_shutdown_pool, app.state.audio_pool)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet TDT 0.6B v3 (CPU)",
        version=__version__,
        description=(
            "OpenAI-compatible CPU ASR service for Parakeet TDT 0.6B v3. "
            "Transcription endpoints require `Authorization: Bearer <API key>`."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(ApiKeyMiddleware)
    app.include_router(public_router)
    app.include_router(router)
    return app


app = create_app()
