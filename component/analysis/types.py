"""Typed contracts between the analysis UI, service, and math layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class AnalysisMethod(Enum):
    """Available analysis methods."""

    STRATIFIED_ESTIMATION = "stratified_estimation"

    @classmethod
    def from_string(cls, value: str) -> "AnalysisMethod":
        for method in cls:
            if method.value == value.lower():
                return method
        raise ValueError(f"Unknown analysis method: {value}")


@dataclass
class AnalysisInputs:
    """Validated, standardized inputs for an analysis run."""

    method: AnalysisMethod
    reference_df: pd.DataFrame  # standardized: map_code, ref_code, area [, x, y, ...]
    area_data: pd.DataFrame  # standardized: map_code, map_area
    confidence_level: float  # percentage, e.g. 95.0
    column_mapping: Dict[str, str]  # role -> source column (kept for export/labels)
    filter_spec: Optional[Dict] = None  # {"column", "include_values"}
    area_unit: str = "ha"  # display unit; math stays native
    class_names: Optional[Dict[int, str]] = None  # map_code -> human label

    @property
    def confidence_level_decimal(self) -> float:
        return self.confidence_level / 100.0


@dataclass
class ClassEstimate:
    map_code: int
    class_name: str
    number_samples: float
    map_pixel_count: float
    area_estimate: float  # error-adjusted stratified (Olofsson Eq. 8)
    standard_error: float
    confidence_interval: float
    srs_area_estimate: float
    srs_standard_error: float
    srs_confidence_interval: float


@dataclass
class AccuracyRow:
    map_code: int
    class_name: str
    users_accuracy: float
    producers_accuracy: float
    weighted_producers_accuracy: float


@dataclass
class AnalysisResults:
    method: AnalysisMethod
    success: bool = True
    error_message: Optional[str] = None
    confusion_matrix: Optional[pd.DataFrame] = None
    class_estimates: List[ClassEstimate] = field(default_factory=list)
    accuracy_rows: List[AccuracyRow] = field(default_factory=list)
    overall_accuracy: float = 0.0
    total_area: float = 0.0  # A_total, native unit
    area_unit: str = "ha"
    confidence_level: float = 0.0
    z: float = 0.0
    map_legend: List = field(default_factory=list)
    ref_legend: List = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        cm = None
        if self.confusion_matrix is not None:
            cm = {
                "index": list(self.confusion_matrix.index),
                "columns": list(self.confusion_matrix.columns),
                "data": self.confusion_matrix.values.tolist(),
            }
        return {
            "method": self.method.value,
            "success": self.success,
            "error_message": self.error_message,
            "confusion_matrix": cm,
            "class_estimates": [vars(c).copy() for c in self.class_estimates],
            "accuracy_rows": [vars(a).copy() for a in self.accuracy_rows],
            "overall_accuracy": self.overall_accuracy,
            "total_area": self.total_area,
            "area_unit": self.area_unit,
            "confidence_level": self.confidence_level,
            "z": self.z,
            "map_legend": list(self.map_legend),
            "ref_legend": list(self.ref_legend),
        }

    @classmethod
    def error(cls, method: AnalysisMethod, message: str) -> "AnalysisResults":
        return cls(method=method, success=False, error_message=message)
