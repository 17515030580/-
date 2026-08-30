# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class ParsedOmics:
    modality: str
    sample_id: Optional[str]
    values: pd.Series
    source_filename: str
    orientation: str


@dataclass
class RawOmicsSample:
    patient_id: str
    expression: ParsedOmics
    mutation: ParsedOmics
    methylation: ParsedOmics

    def by_modality(self) -> Dict[str, ParsedOmics]:
        return {
            "expression": self.expression,
            "mutation": self.mutation,
            "methylation": self.methylation,
        }
