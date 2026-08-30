# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.utils.json_utils import json_safe


class ResultStore:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, prediction_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
        if not self.enabled:
            return {}
        result_dir = self.root / prediction_id
        result_dir.mkdir(parents=True, exist_ok=True)
        safe_payload = json_safe(payload)

        with (result_dir / "prediction_result.json").open("w", encoding="utf-8") as handle:
            json.dump(safe_payload, handle, ensure_ascii=False, indent=2)

        qc = safe_payload.get("quality_control", {})
        with (result_dir / "quality_control.json").open("w", encoding="utf-8") as handle:
            json.dump(qc, handle, ensure_ascii=False, indent=2)

        drug = safe_payload.get("drug_response", {})
        all_drugs = drug.get("all_drugs", []) if isinstance(drug, dict) else []
        top10 = drug.get("top10", []) if isinstance(drug, dict) else []
        if all_drugs:
            pd.DataFrame(all_drugs).to_csv(
                result_dir / "all_drug_predictions.csv", index=False
            )
        if top10:
            pd.DataFrame(top10).to_csv(
                result_dir / "top10_drug_predictions.csv", index=False
            )

        subtype = safe_payload.get("subtype", {})
        probabilities = subtype.get("probabilities", []) if isinstance(subtype, dict) else []
        if probabilities:
            pd.DataFrame(probabilities).to_csv(
                result_dir / "subtype_probabilities.csv", index=False
            )

        manifest = {
            "prediction_id": prediction_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "files": sorted(path.name for path in result_dir.iterdir() if path.is_file()),
        }
        with (result_dir / "download_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return {name: "/results/{}/download/{}".format(prediction_id, name) for name in manifest["files"]}

    def result_path(self, prediction_id: str, filename: str) -> Path:
        safe_name = Path(filename).name
        path = (self.root / prediction_id / safe_name).resolve()
        expected_root = (self.root / prediction_id).resolve()
        if expected_root not in path.parents:
            raise FileNotFoundError(filename)
        return path
