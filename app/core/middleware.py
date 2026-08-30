# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict


class _BodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """ASGI middleware that checks declared and actually received body bytes."""

    def __init__(self, app: Callable[..., Awaitable[Any]], max_body_size: int):
        self.app = app
        self.max_body_size = int(max_body_size)

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_size:
                    await self._send_413(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    raise _BodyTooLarge()
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not response_started:
                await self._send_413(scope, receive, send)
            else:
                raise

    async def _send_413(self, scope, receive, send):
        payload = json.dumps(
            {
                "success": False,
                "error": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": "上传请求超过服务允许的大小。",
                    "details": {"max_request_bytes": self.max_body_size},
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
