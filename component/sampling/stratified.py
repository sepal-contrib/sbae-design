"""Stratified sampling strategy implementation.

Stratified sampling divides the population into non-overlapping subgroups (strata)
and samples from each stratum. This is the most common method for land cover
accuracy assessment as it ensures representation of all classes.
"""

import logging
from typing import List

from component.sampling.base import SamplingStrategy
from component.sampling.types import (
    ClassAllocation,
    SamplingInputs,
    SamplingMethod,
    SamplingResults,
)
from component.scripts.stratified import (
    calculate_openforis_stratified_design,
    calculate_per_class_moe_for_allocation,
)

logger = logging.getLogger("sbae.sampling.stratified")


class StratifiedSamplingStrategy(SamplingStrategy):
    """Strategy for stratified random sampling.

    Stratified sampling is ideal when you have a classification map and want
    to estimate overall accuracy with controlled precision per class.
    """

    @property
    def method(self) -> SamplingMethod:
        return SamplingMethod.STRATIFIED

    @property
    def display_name(self) -> str:
        return "Stratified Random Sampling"

    @property
    def description(self) -> str:
        return (
            "Divide the area into classes (strata) based on a classification map "
            "and sample from each class. Best for accuracy assessment of land cover maps."
        )

    @property
    def requires_classification_map(self) -> bool:
        return True

    @property
    def supports_per_class_allocation(self) -> bool:
        return True

    def validate_inputs(self, inputs: SamplingInputs) -> List[str]:
        """Validate inputs for stratified sampling."""
        errors = self._validate_common_inputs(inputs)

        # Check for required classification data
        if inputs.area_data is None or inputs.area_data.empty:
            errors.append("Classification map with area data is required")

        # Check minimum samples per class
        if inputs.min_samples_per_class < 1:
            errors.append("Minimum samples per class must be at least 1")
        elif inputs.min_samples_per_class > 100:
            errors.append("Minimum samples per class should not exceed 100")

        if not inputs.expected_accuracies:
            errors.append(
                "Stratified sampling requires expected user accuracy per class"
            )
        elif inputs.area_data is not None and not inputs.area_data.empty:
            expected_codes = set(inputs.expected_accuracies)
            missing_codes = [
                int(row["map_code"])
                for _, row in inputs.area_data.iterrows()
                if int(row["map_code"]) not in expected_codes
            ]
            if missing_codes:
                errors.append(
                    "Expected user accuracy is missing for class code(s): "
                    + ", ".join(str(code) for code in missing_codes)
                )

        return errors

    def calculate(self, inputs: SamplingInputs) -> SamplingResults:
        """Calculate stratified sample design."""
        # Validate first
        errors = self.validate_inputs(inputs)
        if errors:
            return SamplingResults.error(self.method, "; ".join(errors))

        try:
            area_df = inputs.area_data
            target_se = inputs.target_error_decimal
            confidence_level = inputs.confidence_level_decimal
            min_samples = inputs.min_samples_per_class
            expected_accuracies = {
                int(code): float(value)
                for code, value in (inputs.expected_accuracies or {}).items()
            }

            design_df = calculate_openforis_stratified_design(
                area_df=area_df,
                expected_accuracies=expected_accuracies,
                target_standard_error=target_se,
                min_samples_per_class=min_samples,
            )
            allocation_dict = {
                int(row["map_code"]): int(row["final"])
                for _, row in design_df.iterrows()
            }

            # Build per-class allocation results
            total_area = area_df["map_area"].sum()
            samples_per_class = []

            for _, row in design_df.iterrows():
                code = int(row["map_code"])
                class_name = row.get("map_edited_class", f"Class {code}")
                samples = int(row["final"])
                area_ha = row["map_area"] / 10000

                samples_per_class.append(
                    ClassAllocation(
                        map_code=code,
                        class_name=class_name,
                        samples=samples,
                        area_ha=area_ha,
                        proportion=float(row["wi"]),
                        expected_accuracy=float(row["eua"]),
                        equal_samples=int(row["equal"]),
                        proportional_samples=int(row["proportional"]),
                        adjusted_samples=int(row["adjusted"]),
                    )
                )

            moe_df = calculate_per_class_moe_for_allocation(
                allocation=allocation_dict,
                area_df=area_df,
                confidence_level=confidence_level,
                expected_accuracies=expected_accuracies,
            )
            moe_by_code = moe_df.set_index("map_code")["moe_percent"]
            for alloc in samples_per_class:
                alloc.moe_percent = float(moe_by_code.loc[alloc.map_code])

            actual_total = sum(allocation_dict.values())

            return SamplingResults(
                sampling_method=self.method,
                success=True,
                total_samples=actual_total,
                target_error=inputs.target_error,
                confidence_level=inputs.confidence_level,
                allocation_method="adjusted_proportional",
                allocation_dict=allocation_dict,
                samples_per_class=samples_per_class,
                total_area_ha=total_area / 10000,
                # Note: MOE for stratified is per-class, not a single value
                current_moe_percent=None,
                current_moe_decimal=None,
            )

        except Exception as e:
            logger.error(f"Error in stratified sampling calculation: {e}")
            return SamplingResults.error(self.method, str(e))
