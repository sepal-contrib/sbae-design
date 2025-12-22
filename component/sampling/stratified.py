"""Stratified sampling strategy implementation.

Stratified sampling divides the population into non-overlapping subgroups (strata)
and samples from each stratum. This is the most common method for land cover
accuracy assessment as it ensures representation of all classes.
"""

import logging
import math
from typing import Dict, List

from component.sampling.base import SamplingStrategy
from component.sampling.types import (
    AllocationMethod,
    ClassAllocation,
    SamplingInputs,
    SamplingMethod,
    SamplingResults,
)
from component.scripts.stratified import (
    allocate_samples_balanced,
    allocate_samples_equal,
    allocate_samples_neyman,
    apply_adjusted_allocation,
    calculate_per_class_moe_for_allocation,
    calculate_stratified_sample_size,
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

        # Validate allocation method
        if inputs.allocation_method == AllocationMethod.NEYMAN:
            if not inputs.expected_accuracies:
                errors.append(
                    "Neyman allocation requires expected accuracies per class"
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
            expected_accuracies = inputs.expected_accuracies or {}

            # Fill in default expected accuracies if missing
            for _, row in area_df.iterrows():
                code = int(row["map_code"])
                if code not in expected_accuracies:
                    expected_accuracies[code] = inputs.expected_accuracy_decimal

            # Calculate total sample size using stratified formula
            n_total = calculate_stratified_sample_size(
                area_df=area_df,
                expected_accuracies=expected_accuracies,
                target_standard_error=target_se,
            )

            # Allocate samples based on method
            allocation_dict = self._allocate_samples(
                area_df=area_df,
                total_samples=n_total,
                allocation_method=inputs.allocation_method,
                expected_accuracies=expected_accuracies,
                min_samples=min_samples,
            )

            # Build per-class allocation results
            total_area = area_df["map_area"].sum()
            samples_per_class = []

            for _, row in area_df.iterrows():
                code = int(row["map_code"])
                class_name = row.get("map_edited_class", f"Class {code}")
                samples = allocation_dict.get(code, min_samples)
                area_ha = row["map_area"] / 10000
                proportion = row["map_area"] / total_area if total_area > 0 else 0

                samples_per_class.append(
                    ClassAllocation(
                        map_code=code,
                        class_name=class_name,
                        samples=samples,
                        area_ha=area_ha,
                        proportion=proportion,
                    )
                )

            # Calculate per-class MOE
            try:
                moe_df = calculate_per_class_moe_for_allocation(
                    allocation=allocation_dict,
                    area_df=area_df,
                    confidence_level=confidence_level,
                )
                for alloc in samples_per_class:
                    moe_row = moe_df[moe_df["map_code"] == alloc.map_code]
                    if not moe_row.empty:
                        alloc.moe_percent = moe_row["moe_percent"].iloc[0]
            except Exception as e:
                logger.warning(f"Could not calculate per-class MOE: {e}")

            actual_total = sum(allocation_dict.values())

            return SamplingResults(
                sampling_method=self.method,
                success=True,
                total_samples=actual_total,
                target_error=inputs.target_error,
                confidence_level=inputs.confidence_level,
                allocation_method=inputs.allocation_method.value,
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

    def _allocate_samples(
        self,
        area_df,
        total_samples: int,
        allocation_method: AllocationMethod,
        expected_accuracies: Dict[int, float],
        min_samples: int,
    ) -> Dict[int, int]:
        """Allocate samples to classes based on allocation method."""
        if allocation_method == AllocationMethod.EQUAL:
            raw_allocation = allocate_samples_equal(area_df, total_samples)
        elif allocation_method == AllocationMethod.NEYMAN:
            raw_allocation = allocate_samples_neyman(
                area_df, expected_accuracies, total_samples
            )
        elif allocation_method == AllocationMethod.BALANCED:
            raw_allocation = allocate_samples_balanced(area_df, total_samples)
        else:  # PROPORTIONAL (default)
            raw_allocation = apply_adjusted_allocation(
                area_df, total_samples, min_samples
            )
            return raw_allocation  # Already returns integers

        # Apply minimum constraints and convert to integers
        final_allocation = {}
        for code, n_float in raw_allocation.items():
            n_int = max(math.ceil(n_float), min_samples)
            final_allocation[int(code)] = n_int

        return final_allocation
