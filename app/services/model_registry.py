# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from app.adapters.mtegdrp_adapter import MTEGDRPAdapter
from app.adapters.subtype_adapter import SubtypeAdapter
from app.config import Settings
from app.core.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Process-level singleton holding both model adapters."""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, settings: Settings):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Settings) -> None:
        if self._initialized:
            return
        self.settings = settings
        self.drug: Optional[MTEGDRPAdapter] = None
        self.subtype: Optional[SubtypeAdapter] = None
        self._initialized = True

    def load_all(self) -> None:
        self.drug = MTEGDRPAdapter(
            artifact_dir=self.settings.drug_artifact_dir,
            class_path=self.settings.drug_class_path,
            device_name=self.settings.drug_device,
            batch_size=self.settings.drug_batch_size,
            strict_feature_match=self.settings.strict_feature_match,
            reject_non_finite=self.settings.reject_non_finite,
        )
        self.subtype = SubtypeAdapter(
            artifact_dir=self.settings.subtype_artifact_dir,
            class_path=self.settings.subtype_class_path,
            device_name=self.settings.subtype_device,
            forward_mode=self.settings.subtype_forward_mode,
            output_mode=self.settings.subtype_output_mode,
            strict_feature_match=self.settings.strict_feature_match,
            reject_non_finite=self.settings.reject_non_finite,
            enabled=self.settings.subtype_enabled,
        )

        errors = []
        if self.settings.drug_enabled:
            try:
                self.drug.load()
                logger.info("Drug response model loaded: %s", self.drug.status())
            except Exception as exc:
                logger.exception("Failed to load drug response model")
                errors.append("drug_response: " + str(exc))
        if self.settings.subtype_enabled:
            try:
                self.subtype.load()
                logger.info("Subtype model loaded: %s", self.subtype.status())
            except Exception as exc:
                logger.exception("Failed to load subtype model")
                errors.append("subtype: " + str(exc))
        else:
            self.subtype.load()
            logger.warning("Subtype model is disabled; placeholder response will be returned.")

        if errors and not self.settings.allow_partial_startup:
            raise RuntimeError("; ".join(errors))

    def clear(self) -> None:
        self.drug = None
        self.subtype = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        return {
            "drug_response": self.drug.status() if self.drug else {"loaded": False},
            "subtype": self.subtype.status() if self.subtype else {"loaded": False},
            "parallel_inference": self.settings.parallel_inference,
            "require_both_models": self.settings.require_both_models,
        }

    def ensure_predictable(self) -> None:
        drug_ready = bool(self.drug and self.drug.loaded)
        if not drug_ready:
            raise ModelUnavailableError(
                "药敏模型不可用。", {"models": self.status()}
            )
        if self.settings.require_both_models and not (self.subtype and self.subtype.loaded):
            raise ModelUnavailableError(
                "当前配置要求两个模型都可用，但亚型模型尚未加载。",
                {"models": self.status()},
            )
