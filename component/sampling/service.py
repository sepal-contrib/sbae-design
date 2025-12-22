"""Sampling service for orchestrating sampling calculations.

This module provides the main entry points for the UI layer to interact
with the sampling strategies.
"""

import logging
from typing import Dict, Type

from component.sampling.base import SamplingStrategy
from component.sampling.simple import SimpleSamplingStrategy
from component.sampling.stratified import StratifiedSamplingStrategy
from component.sampling.systematic import SystematicSamplingStrategy
from component.sampling.types import (
    AllocationMethod,
    SamplingInputs,
    SamplingMethod,
    SamplingResults,
)

logger = logging.getLogger("sbae.sampling.service")

# Registry of available strategies
_STRATEGY_REGISTRY: Dict[SamplingMethod, Type[SamplingStrategy]] = {
    SamplingMethod.STRATIFIED: StratifiedSamplingStrategy,
    SamplingMethod.SIMPLE: SimpleSamplingStrategy,
    SamplingMethod.SYSTEMATIC: SystematicSamplingStrategy,
}

# Cached strategy instances
_strategy_instances: Dict[SamplingMethod, SamplingStrategy] = {}


def get_sampling_strategy(method: SamplingMethod) -> SamplingStrategy:
    """Get the sampling strategy for a given method.

    Args:
        method: The sampling method

    Returns:
        The corresponding SamplingStrategy instance

    Raises:
        ValueError: If the method is not supported
    """
    if method not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported sampling method: {method}")

    # Use cached instance if available
    if method not in _strategy_instances:
        _strategy_instances[method] = _STRATEGY_REGISTRY[method]()

    return _strategy_instances[method]


def get_strategy_from_string(method_str: str) -> SamplingStrategy:
    """Get sampling strategy from string method name.

    Args:
        method_str: String name of sampling method (e.g., "stratified")

    Returns:
        The corresponding SamplingStrategy instance
    """
    method = SamplingMethod.from_string(method_str)
    return get_sampling_strategy(method)


class SamplingService:
    """High-level service for sampling calculations.

    This service provides a simplified interface for the UI layer,
    handling the conversion between app state and strategy inputs.
    """

    @staticmethod
    def create_inputs_from_state(app_state) -> SamplingInputs:
        """Create SamplingInputs from app state.

        Args:
            app_state: The application state object

        Returns:
            SamplingInputs populated from state
        """
        method_str = app_state.sampling_method.value
        method = SamplingMethod.from_string(method_str)

        # Get allocation method for stratified
        allocation_method = AllocationMethod.PROPORTIONAL
        if hasattr(app_state, "stratified_allocation_method"):
            try:
                allocation_method = AllocationMethod.from_string(
                    app_state.stratified_allocation_method.value
                )
            except ValueError:
                pass

        return SamplingInputs(
            sampling_method=method,
            target_error=app_state.target_error.value,
            confidence_level=app_state.confidence_level.value,
            expected_accuracy=app_state.expected_accuracy.value,
            # Stratified-specific
            area_data=(
                app_state.area_data.value
                if hasattr(app_state.area_data, "value")
                else app_state.area_data
            ),
            allocation_method=allocation_method,
            min_samples_per_class=app_state.min_samples_per_class.value,
            expected_accuracies=app_state.expected_user_accuracies.value,
            file_path=(
                app_state.file_path.value if hasattr(app_state, "file_path") else None
            ),
            # Simple/Systematic-specific
            aoi_gdf=app_state.aoi_gdf.value if hasattr(app_state, "aoi_gdf") else None,
            total_samples=(
                int(app_state.simple_total_samples.value)
                if hasattr(app_state, "simple_total_samples")
                else 100
            ),
        )

    @staticmethod
    def calculate(inputs: SamplingInputs) -> SamplingResults:
        """Calculate sample design using the appropriate strategy.

        Args:
            inputs: Sampling inputs

        Returns:
            SamplingResults from the calculation
        """
        strategy = get_sampling_strategy(inputs.sampling_method)
        return strategy.calculate(inputs)

    @staticmethod
    def calculate_from_state(app_state) -> SamplingResults:
        """Calculate sample design directly from app state.

        This is a convenience method that combines create_inputs_from_state
        and calculate into a single call.

        Args:
            app_state: The application state object

        Returns:
            SamplingResults from the calculation
        """
        inputs = SamplingService.create_inputs_from_state(app_state)
        return SamplingService.calculate(inputs)

    @staticmethod
    def is_ready(app_state) -> bool:
        """Check if the app state is ready for calculation.

        Args:
            app_state: The application state object

        Returns:
            True if ready for calculation
        """
        try:
            inputs = SamplingService.create_inputs_from_state(app_state)
            strategy = get_sampling_strategy(inputs.sampling_method)
            return strategy.is_ready(inputs)
        except Exception:
            return False

    @staticmethod
    def get_validation_errors(app_state) -> list:
        """Get validation errors for current state.

        Args:
            app_state: The application state object

        Returns:
            List of validation error messages
        """
        try:
            inputs = SamplingService.create_inputs_from_state(app_state)
            strategy = get_sampling_strategy(inputs.sampling_method)
            return strategy.validate_inputs(inputs)
        except Exception as e:
            return [str(e)]

    @staticmethod
    def get_available_methods() -> list:
        """Get list of available sampling methods.

        Returns:
            List of (method_value, display_name, description) tuples
        """
        methods = []
        for method in SamplingMethod:
            strategy = get_sampling_strategy(method)
            methods.append((method.value, strategy.display_name, strategy.description))
        return methods
