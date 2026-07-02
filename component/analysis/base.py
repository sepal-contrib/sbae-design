"""Base class for analysis strategies (mirrors sampling/base.py)."""

import logging
from abc import ABC, abstractmethod
from typing import List

from component.analysis.types import AnalysisInputs, AnalysisMethod, AnalysisResults

logger = logging.getLogger("sbae.analysis")

ALLOWED_CONFIDENCE = {90.0, 95.0, 99.0}


class AnalysisStrategy(ABC):
    """Abstract base for analysis methods."""

    @property
    @abstractmethod
    def method(self) -> AnalysisMethod: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def validate_inputs(self, inputs: AnalysisInputs) -> List[str]:
        """Return validation error messages (empty if valid)."""
        ...

    @abstractmethod
    def analyze(self, inputs: AnalysisInputs) -> AnalysisResults:
        """Run the analysis; must not raise on user-correctable errors."""
        ...

    def is_ready(self, inputs: AnalysisInputs) -> bool:
        return len(self.validate_inputs(inputs)) == 0

    def _validate_common_inputs(self, inputs: AnalysisInputs) -> List[str]:
        errors: List[str] = []
        if inputs.reference_df is None or inputs.reference_df.empty:
            errors.append("Reference/validation data is required")
        if inputs.area_data is None or inputs.area_data.empty:
            errors.append("Area/strata data is required")
        if inputs.confidence_level not in ALLOWED_CONFIDENCE:
            errors.append("Confidence level must be one of 90, 95, or 99")
        return errors
