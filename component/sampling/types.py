"""Type definitions for sampling strategies.

Contains data classes that define the inputs and outputs for all sampling methods.
This provides a clear contract between the UI layer and the calculation logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd


class SamplingMethod(Enum):
    """Available sampling methods."""

    STRATIFIED = "stratified"
    SIMPLE = "simple"
    SYSTEMATIC = "systematic"

    @classmethod
    def from_string(cls, value: str) -> "SamplingMethod":
        """Convert string to SamplingMethod enum."""
        for method in cls:
            if method.value == value.lower():
                return method
        raise ValueError(f"Unknown sampling method: {value}")


class AllocationMethod(Enum):
    """Allocation methods for stratified sampling."""

    PROPORTIONAL = "proportional"
    EQUAL = "equal"
    NEYMAN = "neyman"
    BALANCED = "balanced"

    @classmethod
    def from_string(cls, value: str) -> "AllocationMethod":
        """Convert string to AllocationMethod enum."""
        for method in cls:
            if method.value == value.lower():
                return method
        raise ValueError(f"Unknown allocation method: {value}")


@dataclass
class SamplingInputs:
    """Input parameters for sampling calculations.

    This is a unified input structure that contains all possible parameters.
    Each strategy will use only the parameters relevant to it.
    """

    # Common parameters
    sampling_method: SamplingMethod
    target_error: float  # As percentage (e.g., 2.0 for 2%)
    confidence_level: float  # As percentage (e.g., 95.0)
    expected_accuracy: float  # As percentage (e.g., 85.0)

    # Stratified-specific
    area_data: Optional[pd.DataFrame] = None
    allocation_method: AllocationMethod = AllocationMethod.PROPORTIONAL
    min_samples_per_class: int = 30
    expected_accuracies: Optional[Dict[int, float]] = None  # Per-class EUA (0-1 scale)
    file_path: Optional[str] = None

    # Simple/Systematic-specific
    aoi_gdf: Optional[gpd.GeoDataFrame] = None
    total_samples: int = 100  # User-specified total for simple/systematic

    def to_decimal(self, percentage: float) -> float:
        """Convert percentage to decimal."""
        return percentage / 100.0

    @property
    def target_error_decimal(self) -> float:
        """Target error as decimal (0-1)."""
        return self.to_decimal(self.target_error)

    @property
    def confidence_level_decimal(self) -> float:
        """Confidence level as decimal (0-1)."""
        return self.to_decimal(self.confidence_level)

    @property
    def expected_accuracy_decimal(self) -> float:
        """Expected accuracy as decimal (0-1)."""
        return self.to_decimal(self.expected_accuracy)


@dataclass
class PrecisionPoint:
    """A single point on the precision curve."""

    sample_size: int
    moe_decimal: float
    moe_percent: float


@dataclass
class ClassAllocation:
    """Sample allocation for a single class (stratified sampling)."""

    map_code: int
    class_name: str
    samples: int
    area_ha: float = 0.0
    proportion: float = 0.0
    moe_percent: Optional[float] = None


@dataclass
class SamplingResults:
    """Results from sampling calculation.

    This is a unified output structure that contains all possible results.
    Some fields may be None depending on the sampling method used.
    """

    # Metadata
    sampling_method: SamplingMethod
    success: bool = True
    error_message: Optional[str] = None

    # Core results
    total_samples: int = 0
    target_error: float = 0.0  # As percentage
    confidence_level: float = 0.0  # As percentage

    # MOE results (for simple/systematic, calculated from formula)
    current_moe_percent: Optional[float] = None
    current_moe_decimal: Optional[float] = None

    # Precision curve (primarily for simple/systematic visualization)
    precision_curve: List[PrecisionPoint] = field(default_factory=list)

    # Stratified-specific results
    allocation_method: Optional[str] = None
    allocation_dict: Dict[int, int] = field(default_factory=dict)
    samples_per_class: List[ClassAllocation] = field(default_factory=list)

    # Area information
    total_area_ha: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for backward compatibility with existing code."""
        result = {
            "sampling_method": self.sampling_method.value,
            "total_samples": self.total_samples,
            "target_error": self.target_error,
            "confidence_level": self.confidence_level,
            "current_moe_percent": self.current_moe_percent,
            "current_moe_decimal": self.current_moe_decimal,
            "allocation_method": self.allocation_method,
            "allocation_dict": self.allocation_dict,
            "total_area_ha": self.total_area_ha,
        }

        # Convert precision curve to list of dicts
        if self.precision_curve:
            result["precision_curve"] = [
                {
                    "sample_size": p.sample_size,
                    "moe_decimal": p.moe_decimal,
                    "moe_percent": p.moe_percent,
                }
                for p in self.precision_curve
            ]
        else:
            result["precision_curve"] = None

        # Convert samples_per_class to list of dicts
        if self.samples_per_class:
            result["samples_per_class"] = [
                {
                    "map_code": c.map_code,
                    "class_name": c.class_name,
                    "samples": c.samples,
                    "area_ha": c.area_ha,
                    "proportion": c.proportion,
                    "moe_percent": c.moe_percent,
                }
                for c in self.samples_per_class
            ]
        else:
            result["samples_per_class"] = []

        return result

    @classmethod
    def error(cls, method: SamplingMethod, message: str) -> "SamplingResults":
        """Create an error result."""
        return cls(
            sampling_method=method,
            success=False,
            error_message=message,
        )
