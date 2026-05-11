"""FastAPI app entry."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import jobs
from .db import init_db
from .routes import router

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

# Tailscale macbook origin (configurable in case the hostname changes)
_macbook_host = os.environ.get("DASHBOARD_TAILSCALE_HOST", "macbook-3.tail-scale.ts.net")
ALLOWED_ORIGINS.extend([
    f"http://{_macbook_host}:3000",
    f"http://{_macbook_host}:3001",
])

ALLOW_ORIGIN_REGEX = r"^https?://([a-z0-9-]+\.)?ngrok(\.io|-free\.app|\.app)$"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await jobs.start_worker()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemmacademy API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=ALLOW_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
