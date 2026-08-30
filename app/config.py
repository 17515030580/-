# -*- coding: utf-8 -*-
"""Environment based configuration compatible with Python 3.8."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _split_csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    app_name: str
    app_version: str
    host: str
    port: int
    log_level: str

    cors_origins: List[str]
    cors_allow_credentials: bool

    max_file_size_bytes: int
    max_request_size_bytes: int
    allowed_extensions: List[str]

    strict_feature_match: bool
    reject_non_finite: bool
    allow_partial_startup: bool
    require_both_models: bool
    parallel_inference: bool
    store_results: bool
    result_root: Path

    drug_enabled: bool
    drug_artifact_dir: Path
    drug_class_path: str
    drug_device: str
    drug_batch_size: int

    subtype_enabled: bool
    subtype_artifact_dir: Path
    subtype_class_path: str
    subtype_device: str
    subtype_forward_mode: str
    subtype_output_mode: str

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])
        ).resolve()
        mb = 1024 * 1024
        return cls(
            project_root=project_root,
            app_name=os.getenv("APP_NAME", "MTEGDRP Multi-omics Prediction API"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_as_int("PORT", 8000),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            cors_origins=_split_csv(
                "CORS_ORIGINS",
                "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
            ),
            cors_allow_credentials=_as_bool("CORS_ALLOW_CREDENTIALS", True),
            max_file_size_bytes=_as_int("MAX_FILE_SIZE_MB", 25) * mb,
            max_request_size_bytes=_as_int("MAX_REQUEST_SIZE_MB", 80) * mb,
            allowed_extensions=_split_csv("ALLOWED_EXTENSIONS", ".csv,.txt"),
            strict_feature_match=_as_bool("STRICT_FEATURE_MATCH", True),
            reject_non_finite=_as_bool("REJECT_NON_FINITE", True),
            allow_partial_startup=_as_bool("ALLOW_PARTIAL_STARTUP", True),
            require_both_models=_as_bool("REQUIRE_BOTH_MODELS", False),
            parallel_inference=_as_bool("PARALLEL_INFERENCE", True),
            store_results=_as_bool("STORE_RESULTS", True),
            result_root=Path(
                os.getenv("RESULT_ROOT", project_root / "runtime_results")
            ).resolve(),
            drug_enabled=_as_bool("DRUG_ENABLED", True),
            drug_artifact_dir=Path(
                os.getenv(
                    "DRUG_ARTIFACT_DIR",
                    project_root / "artifacts" / "drug_response" / "v1.0.0",
                )
            ).resolve(),
            drug_class_path=os.getenv(
                "DRUG_CLASS_PATH", "models.MTEGDRP:MTEGDRP"
            ),
            drug_device=os.getenv("DRUG_DEVICE", "cuda:0"),
            drug_batch_size=_as_int("DRUG_BATCH_SIZE", 64),
            subtype_enabled=_as_bool("SUBTYPE_ENABLED", False),
            subtype_artifact_dir=Path(
                os.getenv(
                    "SUBTYPE_ARTIFACT_DIR",
                    project_root / "artifacts" / "subtype" / "v1.0.0",
                )
            ).resolve(),
            subtype_class_path=os.getenv(
                "SUBTYPE_CLASS_PATH", "subtype_models.model:SubtypeClassifier"
            ),
            subtype_device=os.getenv("SUBTYPE_DEVICE", "cuda:0"),
            subtype_forward_mode=os.getenv("SUBTYPE_FORWARD_MODE", "auto"),
            subtype_output_mode=os.getenv("SUBTYPE_OUTPUT_MODE", "auto"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
