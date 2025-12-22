"""Sampling strategies module.

This module provides a clean, extensible architecture for different sampling methods.
Each sampling method is implemented as a Strategy class that handles:
- Input validation
- Sample size calculation
- Precision/MOE calculation
- Results formatting

Usage:
    from component.sampling import get_sampling_strategy, SamplingMethod

    strategy = get_sampling_strategy(SamplingMethod.STRATIFIED)
    if strategy.validate_inputs(inputs):
        results = strategy.calculate(inputs)
"""

from component.sampling.base import SamplingStrategy
from component.sampling.service import SamplingService, get_sampling_strategy
from component.sampling.types import (
    SamplingInputs,
    SamplingMethod,
    SamplingResults,
)

__all__ = [
    "SamplingStrategy",
    "SamplingMethod",
    "SamplingInputs",
    "SamplingResults",
    "SamplingService",
    "get_sampling_strategy",
]
