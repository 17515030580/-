# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.core.exceptions import PredictionError
from app.schemas.domain import RawOmicsSample
from app.services.model_registry import ModelRegistry
from app.services.result_store import ResultStore
from app.utils.json_utils import json_safe


class PredictionService:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        result_store: ResultStore,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.result_store = result_store

    async def _run_parallel(self, raw: RawOmicsSample):
        subtype_future = run_in_threadpool(self.registry.subtype.predict, raw)
        drug_future = run_in_threadpool(self.registry.drug.predict, raw)
        return await asyncio.gather(subtype_future, drug_future, return_exceptions=True)

    async def _run_serial(self, raw: RawOmicsSample):
        subtype = await run_in_threadpool(self.registry.subtype.predict, raw)
        drug = await run_in_threadpool(self.registry.drug.predict, raw)
        return subtype, drug

    async def predict(self, raw: RawOmicsSample, request_id: str) -> Dict[str, Any]:
        self.registry.ensure_predictable()
        if self.settings.parallel_inference:
            subtype_result, drug_result = await self._run_parallel(raw)
        else:
            subtype_result, drug_result = await self._run_serial(raw)

        errors = {}
        if isinstance(subtype_result, Exception):
            errors["subtype"] = str(subtype_result)
            subtype_result = {"status": "error", "message": str(subtype_result)}
        if isinstance(drug_result, Exception):
            errors["drug_response"] = str(drug_result)
            drug_result = {"status": "error", "message": str(drug_result)}

        if self.settings.require_both_models and errors:
            raise PredictionError("聚合预测失败。", {"model_errors": errors})
        if drug_result.get("status") != "success":
            raise PredictionError("药敏分支预测失败。", {"model_errors": errors})

        prediction_id = "PRED_{}_{}".format(
            datetime.now().strftime("%Y%m%d%H%M%S"), uuid.uuid4().hex[:8]
        )
        quality_control = {
            "input": {
                "patient_id": raw.patient_id,
                "three_omics_complete": True,
                "files": {
                    modality: {
                        "filename": parsed.source_filename,
                        "detected_orientation": parsed.orientation,
                        "uploaded_feature_count": int(len(parsed.values)),
                    }
                    for modality, parsed in raw.by_modality().items()
                },
            },
            "subtype_branch": subtype_result.get("quality_control"),
            "drug_response_branch": drug_result.get("quality_control"),
        }
        payload = json_safe(
            {
                "success": True,
                "request_id": request_id,
                "prediction_id": prediction_id,
                "patient_id": raw.patient_id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "subtype": subtype_result,
                "drug_response": drug_result,
                "quality_control": quality_control,
                "model_status": self.registry.status(),
                "warnings": [
                    "药物排序依据模型预测IC50，不等同于临床处方建议。"
                ],
            }
        )
        downloads = self.result_store.save(prediction_id, payload)
        payload["downloads"] = downloads
        # Save again so the JSON itself also contains download links.
        if downloads:
            self.result_store.save(prediction_id, payload)
        return payload
