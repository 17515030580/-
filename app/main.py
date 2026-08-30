# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestSizeLimitMiddleware
from app.services.model_registry import ModelRegistry
from app.services.prediction_service import PredictionService
from app.services.result_store import ResultStore

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry(settings)
    registry.load_all()
    result_store = ResultStore(settings.result_root, enabled=settings.store_results)
    app.state.registry = registry
    app.state.result_store = result_store
    app.state.prediction_service = PredictionService(
        settings=settings,
        registry=registry,
        result_store=result_store,
    )
    logger.info("Application started. model_status=%s", registry.status())
    yield
    registry.clear()
    logger.info("Application stopped.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="三组学癌症亚型判断与药物敏感性聚合预测服务",
    lifespan=lifespan,
)

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_size=settings.max_request_size_bytes,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


register_exception_handlers(app)
app.include_router(router)
