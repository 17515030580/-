# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from app.schemas.domain import RawOmicsSample


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, raw: RawOmicsSample) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        raise NotImplementedError
