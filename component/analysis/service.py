"""High-level analysis service: adapts app state to strategy inputs."""

import logging
from typing import Dict, Type

import pandas as pd

from component.analysis.base import AnalysisStrategy
from component.analysis.stratified_estimation import StratifiedEstimationStrategy
from component.analysis.types import AnalysisInputs, AnalysisMethod, AnalysisResults
from component.scripts.accuracy import standardize_area, standardize_reference

logger = logging.getLogger("sbae.analysis.service")

_STRATEGY_REGISTRY: Dict[AnalysisMethod, Type[AnalysisStrategy]] = {
    AnalysisMethod.STRATIFIED_ESTIMATION: StratifiedEstimationStrategy,
}
_strategy_instances: Dict[AnalysisMethod, AnalysisStrategy] = {}


def get_analysis_strategy(method: AnalysisMethod) -> AnalysisStrategy:
    if method not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported analysis method: {method}")
    if method not in _strategy_instances:
        _strategy_instances[method] = _STRATEGY_REGISTRY[method]()
    return _strategy_instances[method]


class AnalysisService:
    @staticmethod
    def create_inputs_from_state(app_state) -> AnalysisInputs:
        mapping = dict(app_state.analysis_column_mapping.value or {})
        raw_ref = app_state.analysis_reference_df.value
        ref = (
            standardize_reference(raw_ref, mapping)
            if raw_ref is not None and not raw_ref.empty
            else pd.DataFrame()
        )

        if app_state.analysis_area_source.value == "upload":
            raw_area = app_state.analysis_area_df.value
            area = (
                standardize_area(raw_area, mapping)
                if raw_area is not None and not raw_area.empty
                else pd.DataFrame()
            )
        elif app_state.analysis_area_source.value == "map":
            # raster-derived area table is already canonical map_code / map_area
            raw_area = app_state.analysis_area_df.value
            area = raw_area.copy() if raw_area is not None else pd.DataFrame()
        else:
            # design side already yields map_code / map_area
            area = app_state.area_data.value
            area = area.copy() if area is not None else pd.DataFrame()

        return AnalysisInputs(
            method=AnalysisMethod.STRATIFIED_ESTIMATION,
            reference_df=ref,
            area_data=area,
            confidence_level=float(app_state.analysis_confidence_level.value),
            column_mapping=mapping,
            filter_spec=app_state.analysis_filter.value,
            area_unit=app_state.analysis_area_unit.value,
            class_names=app_state.get_class_lookup(),
        )

    @staticmethod
    def analyze(inputs: AnalysisInputs) -> AnalysisResults:
        return get_analysis_strategy(inputs.method).analyze(inputs)

    @staticmethod
    def analyze_from_state(app_state) -> AnalysisResults:
        return AnalysisService.analyze(
            AnalysisService.create_inputs_from_state(app_state)
        )

    @staticmethod
    def inputs_signature(app_state) -> tuple:
        """A cheap, stable fingerprint of every input that affects the result.

        The Analysis tab computes results only on an explicit Calculate and
        stores this signature alongside them; when the live inputs no longer
        produce the same signature, the shown dashboard is stale and hidden
        until the user recalculates. Uses table shapes + names (not full
        content) to stay cheap while still changing whenever a table is
        loaded/cleared/replaced or the map derivation adds a ``map_code``
        column.
        """
        import json

        def _shape(df):
            if df is None or getattr(df, "empty", True):
                return (0, 0)
            return (int(df.shape[0]), int(df.shape[1]))

        mapping = app_state.analysis_column_mapping.value or {}
        filt = app_state.analysis_filter.value
        return (
            app_state.analysis_area_source.value,
            tuple(sorted((str(k), str(v)) for k, v in mapping.items())),
            float(app_state.analysis_confidence_level.value),
            str(app_state.analysis_area_unit.value),
            json.dumps(filt, sort_keys=True, default=str) if filt else None,
            app_state.analysis_reference_name.value,
            _shape(app_state.analysis_reference_df.value),
            app_state.analysis_area_name.value,
            _shape(app_state.analysis_area_df.value),
            _shape(app_state.area_data.value),
            app_state.analysis_classification_path.value,
        )

    @staticmethod
    def is_ready(app_state) -> bool:
        try:
            inputs = AnalysisService.create_inputs_from_state(app_state)
            return get_analysis_strategy(inputs.method).is_ready(inputs)
        except Exception:
            return False

    @staticmethod
    def get_validation_errors(app_state) -> list:
        try:
            inputs = AnalysisService.create_inputs_from_state(app_state)
            return get_analysis_strategy(inputs.method).validate_inputs(inputs)
        except Exception as e:
            return [str(e)]
