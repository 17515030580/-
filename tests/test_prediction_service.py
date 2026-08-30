# -*- coding: utf-8 -*-
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.schemas.domain import ParsedOmics, RawOmicsSample
from app.services.prediction_service import PredictionService
from app.services.result_store import ResultStore


class _Adapter:
    def __init__(self, result):
        self.result = result

    def predict(self, raw):
        return self.result


class _Registry:
    def __init__(self):
        self.subtype = _Adapter({
            "status": "success",
            "predicted_subtype": "A",
            "probabilities": [{"subtype": "A", "probability": 1.0}],
        })
        self.drug = _Adapter({
            "status": "success",
            "total_drugs": 1,
            "top10": [{"rank": 1, "drug_name": "D", "predicted_ic50_transformed": 0.1}],
            "all_drugs": [{"rank": 1, "drug_name": "D", "predicted_ic50_transformed": 0.1}],
        })

    def ensure_predictable(self):
        return None

    def status(self):
        return {"drug_response": {"loaded": True}, "subtype": {"loaded": True}}


def test_aggregate_prediction(tmp_path: Path):
    parsed = ParsedOmics("expression", "P1", pd.Series({"G": 1.0}), "x.csv", "wide")
    raw = RawOmicsSample("P1", parsed, parsed, parsed)
    settings = SimpleNamespace(
        parallel_inference=True,
        require_both_models=True,
    )
    store = ResultStore(tmp_path, enabled=True)
    service = PredictionService(settings, _Registry(), store)
    payload = asyncio.run(service.predict(raw, "REQ1"))
    assert payload["success"] is True
    assert payload["subtype"]["predicted_subtype"] == "A"
    assert payload["drug_response"]["total_drugs"] == 1
    assert (tmp_path / payload["prediction_id"] / "prediction_result.json").is_file()
