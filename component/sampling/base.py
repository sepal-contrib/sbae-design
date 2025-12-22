"""Base class for sampling strategies.

Defines the interface that all sampling strategies must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from component.sampling.types import SamplingInputs, SamplingMethod, SamplingResults

logger = logging.getLogger("sbae.sampling")


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies.

    Each sampling method (stratified, simple, systematic) implements this interface.
    This ensures consistent behavior and makes it easy to add new methods.
    """

    @property
    @abstractmethod
    def method(self) -> SamplingMethod:
        """Return the sampling method this strategy handles."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this sampling method."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of when to use this sampling method."""
        pass

    @property
    def requires_classification_map(self) -> bool:
        """Whether this method requires a classification map."""
        return False

    @property
    def requires_aoi(self) -> bool:
        """Whether this method requires an AOI selection."""
        return False

    @property
    def supports_precision_curve(self) -> bool:
        """Whether this method supports precision curve visualization."""
        return False

    @property
    def supports_per_class_allocation(self) -> bool:
        """Whether this method supports per-class sample allocation."""
        return False

    @abstractmethod
    def validate_inputs(self, inputs: SamplingInputs) -> List[str]:
        """Validate inputs for this sampling method.

        Args:
            inputs: Sampling inputs to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    @abstractmethod
    def calculate(self, inputs: SamplingInputs) -> SamplingResults:
        """Calculate sample design for this method.

        Args:
            inputs: Validated sampling inputs

        Returns:
            SamplingResults with calculated values
        """
        pass

    def is_ready(self, inputs: SamplingInputs) -> bool:
        """Check if inputs are ready for calculation.

        Args:
            inputs: Sampling inputs to check

        Returns:
            True if ready for calculation
        """
        errors = self.validate_inputs(inputs)
        return len(errors) == 0

    def _validate_common_inputs(self, inputs: SamplingInputs) -> List[str]:
        """Validate inputs common to all sampling methods.

        Args:
            inputs: Sampling inputs to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if inputs.target_error <= 0:
            errors.append("Target error must be greater than 0")
        elif inputs.target_error < 1.0:
            errors.append("Target error must be at least 1%")
        elif inputs.target_error > 10.0:
            errors.append("Target error should not exceed 10%")

        if inputs.confidence_level < 90.0:
            errors.append("Confidence level must be at least 90%")
        elif inputs.confidence_level > 99.0:
            errors.append("Confidence level must not exceed 99%")

        if inputs.expected_accuracy < 50.0:
            errors.append("Expected accuracy must be at least 50%")
        elif inputs.expected_accuracy > 99.0:
            errors.append("Expected accuracy must not exceed 99%")

        return errors
