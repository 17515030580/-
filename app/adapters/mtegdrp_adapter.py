# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.adapters.base import ModelAdapter
from app.core.exceptions import ArtifactError, PredictionError
from app.schemas.domain import RawOmicsSample
from app.services.omics_preprocessor import OmicsPreprocessor, PreprocessedOmics
from app.utils.imports import import_symbol
from app.utils.json_utils import json_safe


class MTEGDRPAdapter(ModelAdapter):
    name = "drug_response"

    def __init__(
        self,
        artifact_dir: Path,
        class_path: str,
        device_name: str,
        batch_size: int,
        strict_feature_match: bool,
        reject_non_finite: bool,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.class_path = class_path
        self.device_name = device_name
        self.batch_size = int(batch_size)
        self.strict_feature_match = strict_feature_match
        self.reject_non_finite = reject_non_finite
        self.loaded = False
        self.load_error = None
        self.model = None
        self.device = None
        self.preprocessor = None
        self.drug_graphs: Dict[str, Any] = {}
        self.catalog = pd.DataFrame()
        self.statistics = pd.DataFrame()
        self.performance = pd.DataFrame()
        self.reliability = pd.DataFrame()
        self.model_manifest: Dict[str, Any] = {}
        self.data_manifest: Dict[str, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

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
            raise ArtifactError("药敏模型权重不是state_dict格式。")
        result = {}
        for key, value in checkpoint.items():
            result[key[7:] if key.startswith("module.") else key] = value
        return result

    def load(self) -> None:
        try:
            import torch

            model_file = self.artifact_dir / "model" / "model_state.pt"
            graph_file = self.artifact_dir / "drugs" / "drug_graphs.pkl"
            catalog_file = self.artifact_dir / "drugs" / "drug_catalog.csv"
            missing = [
                str(path)
                for path in (model_file, graph_file, catalog_file)
                if not path.is_file()
            ]
            if missing:
                raise ArtifactError(
                    "药敏模型部署产物不完整。", {"missing_files": missing}
                )

            self.preprocessor = OmicsPreprocessor(
                self.artifact_dir,
                strict_feature_match=self.strict_feature_match,
                reject_non_finite=self.reject_non_finite,
            )
            self.model_manifest = self._read_json(
                self.artifact_dir / "metadata" / "model_manifest.json"
            )
            self.data_manifest = self._read_json(
                self.artifact_dir / "metadata" / "data_manifest.json"
            )
            class_path = self.model_manifest.get("class_path", self.class_path)
            init_kwargs = self.model_manifest.get("model_init_kwargs", {})
            model_class = import_symbol(class_path)
            self.device = self._resolve_device(torch, self.device_name)
            self.model = model_class(**init_kwargs).to(self.device)
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(self._extract_state_dict(checkpoint), strict=True)
            self.model.eval()

            with graph_file.open("rb") as handle:
                self.drug_graphs = pickle.load(handle)
            self.catalog = self._read_csv(catalog_file)
            self.statistics = self._read_csv(
                self.artifact_dir / "drugs" / "drug_ic50_statistics.csv"
            )
            self.performance = self._read_csv(
                self.artifact_dir / "drugs" / "drug_performance.csv"
            )
            self.reliability = self._read_csv(
                self.artifact_dir / "drugs" / "drug_reliability_reference.csv"
            )

            if self.catalog.empty or not self.drug_graphs:
                raise ArtifactError("药物目录或预计算药物图为空。")
            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)
            raise

    @staticmethod
    def _extract_output(output):
        import torch

        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in (
                "predicted_ic50", "prediction", "predictions", "output", "logits"
            ):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        raise PredictionError("无法从药敏模型输出中识别预测张量。")

    @staticmethod
    def _inverse_ic50(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values.astype(float), 1e-7, 1.0 - 1e-7)
        return 10.0 * np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _percentile(history_text, prediction: float):
        if not isinstance(history_text, str) or not history_text.strip():
            return np.nan
        try:
            history = np.asarray(json.loads(history_text), dtype=float)
            if history.size == 0:
                return np.nan
            return float(np.mean(history <= prediction) * 100.0)
        except Exception:
            return np.nan

    def _build_dataset(self, omics: PreprocessedOmics):
        import torch
        from torch_geometric.data import Data

        ge = torch.tensor(omics.expression[0], dtype=torch.float32)
        mut = torch.tensor(omics.mutation[0], dtype=torch.float32)
        meth = torch.tensor(omics.methylation[0], dtype=torch.float32)
        data_list = []
        ordered_names: List[str] = []

        for row in self.catalog.sort_values("drug_index").itertuples(index=False):
            drug_name = str(row.drug_name)
            graph_entry = self.drug_graphs.get(drug_name)
            if graph_entry is None:
                continue
            graph = graph_entry.get("graph", graph_entry)
            if len(graph) < 4:
                raise ArtifactError(
                    "药物图格式不完整。", {"drug_name": drug_name}
                )
            c_size, features, edge_index, coordinates = graph[:4]
            edge_tensor = torch.tensor(edge_index, dtype=torch.long)
            if edge_tensor.numel() == 0:
                edge_tensor = torch.empty((2, 0), dtype=torch.long)
            elif edge_tensor.ndim == 2 and edge_tensor.shape[1] == 2:
                edge_tensor = edge_tensor.t().contiguous()
            elif edge_tensor.ndim != 2 or edge_tensor.shape[0] != 2:
                raise ArtifactError(
                    "药物edge_index形状错误。", {"drug_name": drug_name}
                )
            item = Data(
                x=torch.tensor(features, dtype=torch.float32),
                edge_index=edge_tensor,
                coordinates=torch.tensor(coordinates, dtype=torch.float32),
                target_ge=ge.unsqueeze(0),
                target_mut=mut.unsqueeze(0),
                target_meth=meth.unsqueeze(0),
                c_size=torch.tensor([int(c_size)], dtype=torch.long),
                smiles=str(graph_entry.get("smiles", getattr(row, "canonical_smiles", ""))),
                drug_name=drug_name,
            )
            data_list.append(item)
            ordered_names.append(drug_name)
        return data_list, ordered_names

    def _merge_metadata(self, drug_names: List[str], transformed: np.ndarray) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "drug_name": drug_names,
                "predicted_ic50_transformed": transformed.astype(float),
                "predicted_ic50_original": self._inverse_ic50(transformed),
            }
        )
        if not self.catalog.empty:
            keep = [
                column for column in (
                    "drug_id", "drug_index", "drug_name", "canonical_smiles",
                    "atom_count", "structure_image"
                ) if column in self.catalog.columns
            ]
            result = result.merge(self.catalog[keep], on="drug_name", how="left")
        if not self.statistics.empty:
            result = result.merge(self.statistics, on="drug_name", how="left")
        if not self.performance.empty:
            result = result.merge(self.performance, on="drug_name", how="left")
        if not self.reliability.empty:
            keep = [
                column for column in (
                    "drug_name", "reference_reliability_level", "reliability_reason"
                ) if column in self.reliability.columns
            ]
            result = result.merge(self.reliability[keep], on="drug_name", how="left")

        percentiles = []
        outside = []
        reliability_levels = []
        for _, row in result.iterrows():
            prediction = float(row["predicted_ic50_transformed"])
            percentiles.append(
                self._percentile(row.get("training_ic50_values"), prediction)
            )
            minimum = row.get("training_ic50_min", np.nan)
            maximum = row.get("training_ic50_max", np.nan)
            is_outside = bool(
                np.isfinite(minimum)
                and np.isfinite(maximum)
                and (prediction < minimum or prediction > maximum)
            )
            outside.append(is_outside)
            static_level = row.get("reference_reliability_level", "unknown")
            if is_outside:
                reliability_levels.append("low")
            else:
                reliability_levels.append(static_level if isinstance(static_level, str) else "unknown")
        result["predicted_percentile"] = percentiles
        result["outside_training_range"] = outside
        result["reliability_level"] = reliability_levels
        result = result.sort_values(
            "predicted_ic50_transformed", ascending=True
        ).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    def predict(self, raw: RawOmicsSample) -> Dict[str, Any]:
        if not self.loaded or self.model is None or self.preprocessor is None:
            raise PredictionError("药敏模型尚未加载。", {"load_error": self.load_error})
        try:
            import torch
            from torch_geometric.loader import DataLoader

            with self._lock:
                omics = self.preprocessor.transform(raw)
                data_list, ordered_names = self._build_dataset(omics)
                loader = DataLoader(
                    data_list,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                predictions = []
                with torch.inference_mode():
                    for batch in loader:
                        batch = batch.to(self.device)
                        output = self._extract_output(self.model(batch))
                        predictions.extend(
                            output.detach().cpu().reshape(-1).numpy().tolist()
                        )
                transformed = np.asarray(predictions, dtype=float)
                if len(transformed) != len(ordered_names):
                    raise PredictionError(
                        "药敏预测数量与药物数量不一致。",
                        {"prediction_count": len(transformed), "drug_count": len(ordered_names)},
                    )
                result = self._merge_metadata(ordered_names, transformed)

            response_columns = [
                column for column in (
                    "rank", "drug_id", "drug_name", "canonical_smiles", "atom_count",
                    "structure_image", "predicted_ic50_transformed",
                    "predicted_ic50_original", "predicted_percentile",
                    "outside_training_range", "reliability_level", "reliability_reason",
                    "training_sample_count", "training_ic50_mean", "training_ic50_median",
                    "training_ic50_min", "training_ic50_max", "test_sample_count",
                    "test_mae", "test_rmse", "test_pearson", "test_spearman", "test_r2"
                ) if column in result.columns
            ]
            compact = result[response_columns]
            return json_safe(
                {
                    "status": "success",
                    "model_name": self.model_manifest.get("model_name", "MTEGDRP"),
                    "model_version": self.model_manifest.get("model_version"),
                    "preprocessing_version": (
                        self.model_manifest.get("preprocessing_version")
                        or self.preprocessor.preprocessing_version
                    ),
                    "device": str(self.device),
                    "total_drugs": int(len(compact)),
                    "ranking_rule": "predicted_ic50_transformed_ascending",
                    "top10": compact.head(10).to_dict(orient="records"),
                    "all_drugs": compact.to_dict(orient="records"),
                    "quality_control": omics.quality_control,
                }
            )
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError("药物敏感性预测失败。", {"error": str(exc)})

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "loaded": self.loaded,
            "enabled": True,
            "artifact_dir": str(self.artifact_dir),
            "device": str(self.device) if self.device is not None else self.device_name,
            "model_version": self.model_manifest.get("model_version"),
            "drug_count": int(len(self.catalog)) if not self.catalog.empty else 0,
            "error": self.load_error,
        }

    def structure_image_path(self, drug_id: str) -> Path:
        if self.catalog.empty or "drug_id" not in self.catalog.columns:
            raise ArtifactError("药物目录中没有drug_id字段。")
        matched = self.catalog[self.catalog["drug_id"].astype(str) == str(drug_id)]
        if matched.empty:
            raise ArtifactError("未找到指定药物。", {"drug_id": drug_id})
        relative = str(matched.iloc[0].get("structure_image", ""))
        path = self.artifact_dir / "drugs" / relative
        if not relative or not path.is_file():
            raise ArtifactError("药物二维结构图不存在。", {"drug_id": drug_id})
        return path
