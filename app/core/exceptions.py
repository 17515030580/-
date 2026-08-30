# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class InputValidationError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("INPUT_VALIDATION_ERROR", message, 422, details)


class FileTooLargeError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("FILE_TOO_LARGE", message, 413, details)


class ArtifactError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("ARTIFACT_ERROR", message, 500, details)


class ModelUnavailableError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("MODEL_UNAVAILABLE", message, 503, details)


class PredictionError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("PREDICTION_ERROR", message, 500, details)
