# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from app.adapters.base import ModelAdapter
from app.core.exceptions import ArtifactError, PredictionError
from app.schemas.domain import RawOmicsSample
from app.services.omics_preprocessor import OmicsPreprocessor
from app.utils.imports import import_symbol
from app.utils.json_utils import json_safe


class SubtypeAdapter(ModelAdapter):
    """Generic adapter reserved for the future cancer subtype model."""

    name = "subtype"

    def __init__(
        self,
        artifact_dir: Path,
        class_path: str,
        device_name: str,
        forward_mode: str,
        output_mode: str,
        strict_feature_match: bool,
        reject_non_finite: bool,
        enabled: bool,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.class_path = class_path
        self.device_name = device_name
        self.forward_mode = forward_mode
        self.output_mode = output_mode
        self.strict_feature_match = strict_feature_match
        self.reject_non_finite = reject_non_finite
        self.enabled = enabled
        self.loaded = False
        self.load_error = None
        self.model = None
        self.device = None
        self.preprocessor = None
        self.class_names: List[str] = []
        self.manifest: Dict[str, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _read_json(path: Path):
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _resolve_device(torch, requested: str):
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break
        if not isinstance(checkpoint, dict):
            raise ArtifactError("亚型模型权重不是state_dict格式。")
        return {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }

    @staticmethod
    def _load_class_names(artifact_dir: Path) -> List[str]:
        candidates = [
            artifact_dir / "metadata" / "subtype_label_map.json",
            artifact_dir / "metadata" / "label_map.json",
            artifact_dir / "subtype_label_map.json",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return [str(item) for item in payload]
            if isinstance(payload, dict) and isinstance(payload.get("classes"), list):
                return [str(item) for item in payload["classes"]]
            if isinstance(payload, dict):
                try:
                    return [str(payload[str(index)]) for index in range(len(payload))]
                except Exception:
                    ordered = sorted(payload.items(), key=lambda item: int(item[0]))
                    return [str(value) for _, value in ordered]
        raise ArtifactError("未找到亚型标签映射文件。")

    def load(self) -> None:
        if not self.enabled:
            self.loaded = False
            self.load_error = "SUBTYPE_ENABLED=false，等待补充亚型模型。"
            return
        try:
            import torch

            model_file = self.artifact_dir / "model" / "model_state.pt"
            if not model_file.is_file():
                raise ArtifactError(
                    "亚型模型权重不存在。", {"missing_file": str(model_file)}
                )
            self.preprocessor = OmicsPreprocessor(
                self.artifact_dir,
                strict_feature_match=self.strict_feature_match,
                reject_non_finite=self.reject_non_finite,
            )
            self.manifest = self._read_json(
                self.artifact_dir / "metadata" / "model_manifest.json"
            )
            class_path = self.manifest.get("class_path", self.class_path)
            init_kwargs = self.manifest.get("model_init_kwargs", {})
            self.forward_mode = self.manifest.get("forward_mode", self.forward_mode)
            self.output_mode = self.manifest.get("output_mode", self.output_mode)
            model_class = import_symbol(class_path)
            self.device = self._resolve_device(torch, self.device_name)
            self.model = model_class(**init_kwargs).to(self.device)
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(self._extract_state_dict(checkpoint), strict=True)
            self.model.eval()
            self.class_names = self._load_class_names(self.artifact_dir)
            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)
            raise

    @staticmethod
    def _extract_tensor(output):
        import torch

        if torch.is_tensor(output):
            return output, None
        if isinstance(output, dict):
            for key in ("probabilities", "probs", "logits", "output", "prediction"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value, key
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item, None
        raise PredictionError("无法从亚型模型输出中识别张量。")

    def _call_model(self, expression, mutation, methylation):
        mode = self.forward_mode.lower()
        if mode == "three_args":
            return self.model(expression, mutation, methylation)
        if mode == "keyword_args":
            return self.model(
                expression=expression, mutation=mutation, methylation=methylation
            )
        if mode == "concat":
            import torch
            return self.model(torch.cat([expression, mutation, methylation], dim=1))
        if mode == "dict":
            return self.model(
                {"expression": expression, "mutation": mutation, "methylation": methylation}
            )
        if mode == "data_object":
            return self.model(
                SimpleNamespace(
                    target_ge=expression,
                    target_mut=mutation,
                    target_meth=methylation,
                )
            )
        if mode != "auto":
            raise ArtifactError("未知的SUBTYPE_FORWARD_MODE。", {"mode": mode})

        parameters = [
            item for item in inspect.signature(self.model.forward).parameters.values()
            if item.name != "self"
        ]
        if len(parameters) >= 3:
            return self.model(expression, mutation, methylation)
        # For one-input models, try the most common multi-omics forms in order.
        errors = []
        import torch
        candidates = [
            lambda: self.model(torch.cat([expression, mutation, methylation], dim=1)),
            lambda: self.model(
                {"expression": expression, "mutation": mutation, "methylation": methylation}
            ),
            lambda: self.model(
                SimpleNamespace(
                    target_ge=expression,
                    target_mut=mutation,
                    target_meth=methylation,
                )
            ),
        ]
        for candidate in candidates:
            try:
                return candidate()
            except (TypeError, AttributeError, KeyError, RuntimeError) as exc:
                errors.append(str(exc))
        raise PredictionError(
            "无法自动匹配亚型模型forward输入形式，请配置SUBTYPE_FORWARD_MODE。",
            {"attempt_errors": errors},
        )

    def predict(self, raw: RawOmicsSample) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "status": "pending_model",
                "message": "亚型模型位置已预留，设置SUBTYPE_ENABLED=true并补齐产物后启用。",
            }
        if not self.loaded or self.model is None or self.preprocessor is None:
            raise PredictionError("亚型模型尚未加载。", {"load_error": self.load_error})
        try:
            import torch

            with self._lock:
                omics = self.preprocessor.transform(raw)
                expression = torch.tensor(omics.expression, dtype=torch.float32, device=self.device)
                mutation = torch.tensor(omics.mutation, dtype=torch.float32, device=self.device)
                methylation = torch.tensor(omics.methylation, dtype=torch.float32, device=self.device)
                with torch.inference_mode():
                    raw_output = self._call_model(expression, mutation, methylation)
                    tensor, detected_key = self._extract_tensor(raw_output)
                    tensor = tensor.detach().float().reshape(1, -1)
                    mode = self.output_mode.lower()
                    if mode == "probabilities" or detected_key in ("probabilities", "probs"):
                        probabilities = tensor
                    elif mode == "logits":
                        probabilities = torch.softmax(tensor, dim=1)
                    elif mode == "auto":
                        row = tensor[0]
                        looks_like_probability = bool(
                            torch.all(row >= 0)
                            and torch.all(row <= 1)
                            and torch.isclose(row.sum(), torch.tensor(1.0, device=row.device), atol=1e-3)
                        )
                        probabilities = tensor if looks_like_probability else torch.softmax(tensor, dim=1)
                    else:
                        raise ArtifactError("未知的SUBTYPE_OUTPUT_MODE。", {"mode": mode})
                    probs = probabilities.cpu().numpy()[0]

            if len(probs) != len(self.class_names):
                raise PredictionError(
                    "亚型输出类别数与标签映射不一致。",
                    {"output_count": len(probs), "label_count": len(self.class_names)},
                )
            order = np.argsort(-probs)
            top1 = int(order[0])
            top2 = int(order[1]) if len(order) > 1 else top1
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            class_rows = [
                {"subtype": self.class_names[index], "probability": float(probs[index])}
                for index in order
            ]
            return json_safe(
                {
                    "status": "success",
                    "model_name": self.manifest.get("model_name", "SubtypeClassifier"),
                    "model_version": self.manifest.get("model_version"),
                    "preprocessing_version": (
                        self.manifest.get("preprocessing_version")
                        or self.preprocessor.preprocessing_version
                    ),
                    "device": str(self.device),
                    "predicted_subtype": self.class_names[top1],
                    "top1_probability": float(probs[top1]),
                    "top2_subtype": self.class_names[top2],
                    "top2_probability": float(probs[top2]),
                    "probability_margin": float(probs[top1] - probs[top2]),
                    "prediction_entropy": entropy,
                    "probabilities": class_rows,
                    "quality_control": omics.quality_control,
                }
            )
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError("癌症亚型预测失败。", {"error": str(exc)})

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "artifact_dir": str(self.artifact_dir),
            "device": str(self.device) if self.device is not None else self.device_name,
            "model_version": self.manifest.get("model_version"),
            "class_count": len(self.class_names),
            "forward_mode": self.forward_mode,
            "output_mode": self.output_mode,
            "error": self.load_error,
        }
