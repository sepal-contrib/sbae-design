"""Accuracy-assessment analysis package (area estimation + accuracies)."""

from component.analysis.base import AnalysisStrategy
from component.analysis.service import AnalysisService, get_analysis_strategy
from component.analysis.stratified_estimation import StratifiedEstimationStrategy
from component.analysis.types import (
    AccuracyRow,
    AnalysisInputs,
    AnalysisMethod,
    AnalysisResults,
    ClassEstimate,
)

__all__ = [
    "AccuracyRow",
    "AnalysisInputs",
    "AnalysisMethod",
    "AnalysisResults",
    "AnalysisService",
    "AnalysisStrategy",
    "ClassEstimate",
    "StratifiedEstimationStrategy",
    "get_analysis_strategy",
]
