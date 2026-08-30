# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _payload(request: Request, code: str, message: str, details: Any = None) -> Dict[str, Any]:
    return {
        "success": False,
        "request_id": _request_id(request),
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(request, exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_payload(
                request,
                "REQUEST_VALIDATION_ERROR",
                "请求参数或上传字段不符合接口要求。",
                {"errors": exc.errors()},
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(request, "HTTP_ERROR", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error. request_id=%s", _request_id(request))
        return JSONResponse(
            status_code=500,
            content=_payload(
                request,
                "INTERNAL_SERVER_ERROR",
                "服务内部错误，请结合request_id检查后端日志。",
            ),
        )
