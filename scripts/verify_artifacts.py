# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings


def check(root: Path, subtype: bool = False):
    required = [
        root / "model" / "model_state.pt",
        root / "preprocessing" / "expression_features.json",
        root / "preprocessing" / "mutation_features.json",
        root / "preprocessing" / "methylation_features.json",
        root / "preprocessing" / "expression_kpca.pkl",
        root / "preprocessing" / "mutation_kpca.pkl",
        root / "preprocessing" / "methylation_kpca.pkl",
    ]
    if not subtype:
        required.extend([
            root / "drugs" / "drug_catalog.csv",
            root / "drugs" / "drug_graphs.pkl",
        ])
    else:
        required.append(root / "metadata" / "subtype_label_map.json")
    return [str(path) for path in required if not path.is_file()]


if __name__ == "__main__":
    settings = get_settings()
    report = {
        "drug_artifact_dir": str(settings.drug_artifact_dir),
        "drug_missing": check(settings.drug_artifact_dir, subtype=False),
        "subtype_artifact_dir": str(settings.subtype_artifact_dir),
        "subtype_enabled": settings.subtype_enabled,
        "subtype_missing": check(settings.subtype_artifact_dir, subtype=True),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
