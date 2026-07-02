"""Stratified error-adjusted area estimation (Olofsson et al. 2014)."""

import logging
from typing import List

import pandas as pd

from component.analysis.base import AnalysisStrategy
from component.analysis.types import (
    AccuracyRow,
    AnalysisInputs,
    AnalysisMethod,
    AnalysisResults,
    ClassEstimate,
)
from component.scripts.accuracy import (
    accuracies_from_matrices,
    apply_filter,
    confusion_matrix_area,
    legends,
    overall_accuracy,
)
from component.scripts.area_estimation import (
    pij_matrix,
    srs_estimates,
    stratified_area_estimates,
    stratum_weights,
)
from component.scripts.calc_utils import get_z_score

logger = logging.getLogger("sbae.analysis.stratified")


class StratifiedEstimationStrategy(AnalysisStrategy):
    @property
    def method(self) -> AnalysisMethod:
        return AnalysisMethod.STRATIFIED_ESTIMATION

    @property
    def display_name(self) -> str:
        return "Stratified area estimation (Olofsson 2014)"

    @property
    def description(self) -> str:
        return (
            "Bias-adjusted area estimates with confidence intervals and "
            "user's/producer's accuracy from a stratified reference sample."
        )

    def validate_inputs(self, inputs: AnalysisInputs) -> List[str]:
        errors = self._validate_common_inputs(inputs)
        if errors:
            return errors

        ref = inputs.reference_df
        for col in ("map_code", "ref_code"):
            if col not in ref.columns:
                errors.append(f"Reference data is missing the '{col}' column mapping")
        area = inputs.area_data
        if "map_area" not in area.columns:
            errors.append("Area data is missing the area-value column mapping")
        elif area["map_area"].le(0).any() or area["map_area"].isna().any():
            errors.append("Area data contains non-positive or non-numeric areas")
        if errors:
            return errors

        ref_map_classes = set(ref["map_code"].dropna().unique().tolist())
        area_classes = set(area["map_code"].dropna().unique().tolist())
        missing = sorted(ref_map_classes - area_classes)
        if missing:
            errors.append(
                "Map classes present in the reference file but missing from the "
                f"area file: {missing}"
            )
        return errors

    def analyze(self, inputs: AnalysisInputs) -> AnalysisResults:
        errors = self.validate_inputs(inputs)
        if errors:
            return AnalysisResults.error(self.method, "; ".join(errors))

        try:
            ref = apply_filter(inputs.reference_df, inputs.filter_spec)
            area = inputs.area_data
            names = inputs.class_names or {}
            z = get_z_score(inputs.confidence_level_decimal)

            map_legend, ref_legend = legends(ref)
            matrix = confusion_matrix_area(ref, map_legend, ref_legend)
            w, a_total = stratum_weights(area)

            # Defensive guard (spec section 6, bug #1): if aligning weights to the
            # map legend produces NaNs, fail loudly instead of letting pij_matrix's
            # fillna(0) silently zero-weight a whole stratum. In practice
            # validate_inputs() already rejects reference map classes missing from
            # the area file, so this should be unreachable via the UI; it's kept
            # as a last-resort check in case callers invoke analyze() directly.
            w_aligned = w.reindex(map_legend)
            if w_aligned.isna().any():
                missing = [c for c in map_legend if bool(pd.isna(w_aligned.get(c)))]
                return AnalysisResults.error(
                    self.method,
                    f"No matching area for map class(es) {missing}; check that reference "
                    "and area class codes have the same type.",
                )

            pij = pij_matrix(matrix, w)

            est = stratified_area_estimates(pij, matrix, a_total, z)  # index=ref_legend
            srs = srs_estimates(matrix, a_total, z)  # index=ref_legend
            acc = accuracies_from_matrices(matrix, pij)  # index=union of classes
            oa = overall_accuracy(pij)

            row_sums = matrix.sum(axis=1)  # per map class
            pixel_by_class = area.groupby("map_code")["map_area"].sum()

            all_classes = sorted(set(map_legend) | set(ref_legend))
            class_estimates = []
            accuracy_rows = []
            for code in all_classes:
                name = names.get(code, f"Class {code}")
                class_estimates.append(
                    _class_estimate(code, name, row_sums, pixel_by_class, est, srs)
                )
                if code in acc.index:
                    a = acc.loc[code]
                    accuracy_rows.append(
                        AccuracyRow(
                            map_code=int(code),
                            class_name=name,
                            users_accuracy=float(a["users_accuracy"]),
                            producers_accuracy=float(a["producers_accuracy"]),
                            weighted_producers_accuracy=float(
                                a["weighted_producers_accuracy"]
                            ),
                        )
                    )

            return AnalysisResults(
                method=self.method,
                success=True,
                confusion_matrix=matrix,
                class_estimates=class_estimates,
                accuracy_rows=accuracy_rows,
                overall_accuracy=float(oa),
                total_area=float(a_total),
                area_unit=inputs.area_unit,
                confidence_level=inputs.confidence_level,
                z=float(z),
                map_legend=map_legend,
                ref_legend=ref_legend,
            )
        except Exception as e:  # unexpected -> error result, never raise to UI
            logger.error("Error in stratified area estimation: %s", e)
            return AnalysisResults.error(self.method, str(e))


def _f(series, code) -> float:
    """Safe float lookup by label, 0.0 if absent."""
    try:
        return float(series.loc[code])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _class_estimate(code, name, row_sums, pixel_by_class, est, srs):
    return ClassEstimate(
        map_code=int(code),
        class_name=name,
        number_samples=_f(row_sums, code),
        map_pixel_count=_f(pixel_by_class, code),
        area_estimate=_f(est["area_estimate"], code),
        standard_error=_f(est["standard_error"], code),
        confidence_interval=_f(est["confidence_interval"], code),
        srs_area_estimate=_f(srs["srs_area_estimate"], code),
        srs_standard_error=_f(srs["srs_standard_error"], code),
        srs_confidence_interval=_f(srs["srs_confidence_interval"], code),
    )
