# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import UploadFile

from app.core.exceptions import FileTooLargeError, InputValidationError


async def read_upload_limited(
    upload: UploadFile,
    max_bytes: int,
    allowed_extensions: List[str],
) -> bytes:
    filename = upload.filename or "unnamed.csv"
    suffix = Path(filename).suffix.lower()
    if suffix not in {item.lower() for item in allowed_extensions}:
        raise InputValidationError(
            "仅允许上传CSV或文本格式的组学矩阵。",
            {"filename": filename, "allowed_extensions": allowed_extensions},
        )

    chunks = []
    total = 0
    await upload.seek(0)
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError(
                "单个上传文件超过大小限制。",
                {"filename": filename, "max_file_bytes": max_bytes},
            )
        chunks.append(chunk)
    await upload.seek(0)
    if total == 0:
        raise InputValidationError("上传文件为空。", {"filename": filename})
    return b"".join(chunks)
