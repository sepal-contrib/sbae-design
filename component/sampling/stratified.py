"""Stratified sampling strategy implementation.

Stratified sampling divides the population into non-overlapping subgroups (strata)
and samples from each stratum. This is the most common method for land cover
accuracy assessment as it ensures representation of all classes.
"""

import logging
from typing import List

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
    allocate_samples_proportional,
    calculate_openforis_stratified_design,
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

        # Per-class EUA is required ONLY for Neyman allocation; proportional /
        # equal / balanced runs use the visible global expected accuracy
        # (AGENTS.md: "Per-class EUA is only active for neyman allocation").
        if inputs.allocation_method == AllocationMethod.NEYMAN:
            if not inputs.expected_accuracies:
                errors.append(
                    "Neyman allocation requires expected user accuracy per class"
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
        """Calculate the stratified sample design honoring the allocation method.

        Neyman allocation uses per-class expected accuracy (the Open Foris /
        Olofsson adjusted-proportional design); proportional / equal / balanced
        use the global expected accuracy for sizing and MOE.
        """
        errors = self.validate_inputs(inputs)
        if errors:
            return SamplingResults.error(self.method, "; ".join(errors))

        try:
            if inputs.allocation_method == AllocationMethod.NEYMAN:
                return self._calculate_neyman(inputs)
            return self._calculate_global_eua(inputs)
        except Exception as e:
            logger.error(f"Error in stratified sampling calculation: {e}")
            return SamplingResults.error(self.method, str(e))

    def _calculate_neyman(self, inputs: SamplingInputs) -> SamplingResults:
        """Per-class EUA path: Olofsson (Open Foris) adjusted-proportional design."""
        area_df = inputs.area_data
        expected_accuracies = {
            int(code): float(value)
            for code, value in (inputs.expected_accuracies or {}).items()
        }
        design_df = calculate_openforis_stratified_design(
            area_df=area_df,
            expected_accuracies=expected_accuracies,
            target_standard_error=inputs.target_error_decimal,
            min_samples_per_class=inputs.min_samples_per_class,
        )
        allocation_dict = {
            int(row["map_code"]): int(row["final"]) for _, row in design_df.iterrows()
        }
        samples_per_class = [
            ClassAllocation(
                map_code=int(row["map_code"]),
                class_name=row.get("map_edited_class", f"Class {int(row['map_code'])}"),
                samples=int(row["final"]),
                area_ha=row["map_area"] / 10000,
                proportion=float(row["wi"]),
                expected_accuracy=float(row["eua"]),
                equal_samples=int(row["equal"]),
                proportional_samples=int(row["proportional"]),
                adjusted_samples=int(row["adjusted"]),
            )
            for _, row in design_df.iterrows()
        ]
        self._attach_moe(
            samples_per_class,
            allocation_dict,
            area_df,
            inputs.confidence_level_decimal,
            expected_accuracies,
        )
        return self._results(
            inputs, allocation_dict, samples_per_class, AllocationMethod.NEYMAN
        )

    def _calculate_global_eua(self, inputs: SamplingInputs) -> SamplingResults:
        """Proportional / equal / balanced using the global expected accuracy."""
        area_df = inputs.area_data
        method = inputs.allocation_method
        min_samples = inputs.min_samples_per_class
        global_eua = inputs.expected_accuracy_decimal
        codes = [int(c) for c in area_df["map_code"]]
        # Uniform (global) EUA -> total sample size ignores per-class EUA.
        eua_dict = {code: global_eua for code in codes}

        total = calculate_stratified_sample_size(
            area_df, eua_dict, inputs.target_error_decimal
        )

        equal_by_code = {
            int(c): v for c, v in allocate_samples_equal(area_df, total).items()
        }
        prop_series = allocate_samples_proportional(area_df, total)
        prop_by_code = {
            int(area_df.iloc[i]["map_code"]): float(prop_series.iloc[i])
            for i in range(len(area_df))
        }
        if method == AllocationMethod.EQUAL:
            raw_by_code = equal_by_code
        elif method == AllocationMethod.BALANCED:
            raw_by_code = {
                int(c): v for c, v in allocate_samples_balanced(area_df, total).items()
            }
        else:  # PROPORTIONAL (default)
            raw_by_code = prop_by_code

        allocation_dict = {
            code: max(round(float(raw_by_code[code])), min_samples) for code in codes
        }

        total_area = area_df["map_area"].sum()
        samples_per_class = []
        for _, row in area_df.iterrows():
            code = int(row["map_code"])
            samples_per_class.append(
                ClassAllocation(
                    map_code=code,
                    class_name=row.get("map_edited_class", f"Class {code}"),
                    samples=allocation_dict[code],
                    area_ha=row["map_area"] / 10000,
                    proportion=float(row["map_area"] / total_area),
                    expected_accuracy=global_eua,
                    equal_samples=round(float(equal_by_code[code])),
                    proportional_samples=round(float(prop_by_code[code])),
                    adjusted_samples=allocation_dict[code],
                )
            )
        self._attach_moe(
            samples_per_class,
            allocation_dict,
            area_df,
            inputs.confidence_level_decimal,
            eua_dict,
        )
        return self._results(inputs, allocation_dict, samples_per_class, method)

    def _attach_moe(self, samples_per_class, allocation_dict, area_df, conf, eua):
        moe_df = calculate_per_class_moe_for_allocation(
            allocation=allocation_dict,
            area_df=area_df,
            confidence_level=conf,
            expected_accuracies=eua,
        )
        moe_by_code = moe_df.set_index("map_code")["moe_percent"]
        for alloc in samples_per_class:
            alloc.moe_percent = float(moe_by_code.loc[alloc.map_code])

    def _results(self, inputs, allocation_dict, samples_per_class, method):
        total_area = inputs.area_data["map_area"].sum()
        return SamplingResults(
            sampling_method=self.method,
            success=True,
            total_samples=sum(allocation_dict.values()),
            target_error=inputs.target_error,
            confidence_level=inputs.confidence_level,
            allocation_method=method.value,
            allocation_dict=allocation_dict,
            samples_per_class=samples_per_class,
            total_area_ha=total_area / 10000,
            current_moe_percent=None,
            current_moe_decimal=None,
        )
