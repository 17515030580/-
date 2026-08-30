# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.core.exceptions import ArtifactError, InputValidationError
from app.services.input_parser import build_raw_sample, parse_omics_csv
from app.services.upload_service import read_upload_limited

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    registry = request.app.state.registry
    status = registry.status()
    drug_ready = status["drug_response"].get("loaded", False)
    subtype_ready = status["subtype"].get("loaded", False)
    return {
        "success": True,
        "status": "ready" if drug_ready else "degraded",
        "drug_response_ready": drug_ready,
        "subtype_ready": subtype_ready,
        "models": status,
    }


@router.get("/models/status")
async def model_status(request: Request):
    return {"success": True, "models": request.app.state.registry.status()}


@router.post("/predict")
async def predict(
    request: Request,
    expression_file: UploadFile = File(..., description="基因表达CSV"),
    mutation_file: UploadFile = File(..., description="突变CSV"),
    methylation_file: UploadFile = File(..., description="甲基化CSV"),
    patient_id: Optional[str] = Form(None),
):
    settings = get_settings()
    uploads = {
        "expression": expression_file,
        "mutation": mutation_file,
        "methylation": methylation_file,
    }
    contents = {}
    for modality, upload in uploads.items():
        contents[modality] = await read_upload_limited(
            upload,
            settings.max_file_size_bytes,
            settings.allowed_extensions,
        )

    parsed = {
        modality: parse_omics_csv(
            contents[modality],
            uploads[modality].filename or (modality + ".csv"),
            modality,
            requested_patient_id=patient_id,
        )
        for modality in uploads
    }
    raw = build_raw_sample(
        expression=parsed["expression"],
        mutation=parsed["mutation"],
        methylation=parsed["methylation"],
        requested_patient_id=patient_id,
    )
    return await request.app.state.prediction_service.predict(
        raw, request.state.request_id
    )


@router.get("/results/{prediction_id}/download/{filename}")
async def download_result(request: Request, prediction_id: str, filename: str):
    path = request.app.state.result_store.result_path(prediction_id, filename)
    if not path.is_file():
        raise InputValidationError(
            "指定的结果文件不存在。",
            {"prediction_id": prediction_id, "filename": filename},
        )
    return FileResponse(path=str(path), filename=path.name)


@router.get("/drugs/{drug_id}/structure")
async def drug_structure(request: Request, drug_id: str):
    registry = request.app.state.registry
    if registry.drug is None or not registry.drug.loaded:
        raise ArtifactError("药敏模型或药物目录尚未加载。")
    path = registry.drug.structure_image_path(drug_id)
    return FileResponse(path=str(path), media_type="image/svg+xml", filename=path.name)
