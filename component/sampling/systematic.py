"""Systematic sampling strategy implementation.

Systematic sampling places sample points on a regular grid pattern across
the AOI. This provides even spatial coverage and is often more practical
for field work than pure random sampling.
"""

import logging
from typing import List

from component.sampling.base import SamplingStrategy
from component.sampling.types import (
    PrecisionPoint,
    SamplingInputs,
    SamplingMethod,
    SamplingResults,
)
from component.scripts.precision import calculate_current_moe, calculate_precision_curve

logger = logging.getLogger("sbae.sampling.systematic")


class SystematicSamplingStrategy(SamplingStrategy):
    """Strategy for systematic sampling.

    Systematic sampling is ideal when:
    - You want even spatial coverage across the AOI
    - Field logistics benefit from regular point spacing
    - You want to detect spatial patterns or gradients
    """

    @property
    def method(self) -> SamplingMethod:
        return SamplingMethod.SYSTEMATIC

    @property
    def display_name(self) -> str:
        return "Systematic Sampling"

    @property
    def description(self) -> str:
        return (
            "Place sample points on a regular grid pattern across the AOI. "
            "Provides even spatial coverage and is practical for field work."
        )

    @property
    def requires_aoi(self) -> bool:
        return True

    @property
    def supports_precision_curve(self) -> bool:
        return True

    def validate_inputs(self, inputs: SamplingInputs) -> List[str]:
        """Validate inputs for systematic sampling."""
        errors = self._validate_common_inputs(inputs)

        # Check for required AOI
        if inputs.aoi_gdf is None:
            errors.append("Area of Interest (AOI) selection is required")

        # Check total samples
        if inputs.total_samples < 10:
            errors.append("Total samples must be at least 10")
        elif inputs.total_samples > 100000:
            errors.append("Total samples should not exceed 100,000")

        return errors

    def calculate(self, inputs: SamplingInputs) -> SamplingResults:
        """Calculate systematic sample design.

        Note: For systematic sampling, we use the same MOE formulas as simple
        random sampling. In practice, systematic sampling often achieves
        similar or better precision, but the theoretical treatment is more
        complex and depends on spatial autocorrelation.
        """
        # Validate first
        errors = self.validate_inputs(inputs)
        if errors:
            return SamplingResults.error(self.method, "; ".join(errors))

        try:
            total_samples = inputs.total_samples
            target_oa = inputs.expected_accuracy_decimal
            confidence_level = inputs.confidence_level_decimal

            # Calculate current MOE (using SRS formula as approximation)
            current_moe = calculate_current_moe(
                current_sample_size=total_samples,
                target_oa=target_oa,
                confidence_level=confidence_level,
            )

            # Generate precision curve
            precision_curve_df = calculate_precision_curve(
                target_oa=target_oa,
                confidence_level=confidence_level,
                min_sample_size=30,
                max_sample_size=max(1000, int(total_samples * 2)),
                num_points=50,
            )

            precision_curve = [
                PrecisionPoint(
                    sample_size=int(row["sample_size"]),
                    moe_decimal=row["moe_decimal"],
                    moe_percent=row["moe_percent"],
                )
                for _, row in precision_curve_df.iterrows()
            ]

            # Calculate AOI area
            total_area_ha = 0.0
            if inputs.aoi_gdf is not None:
                try:
                    total_area_ha = inputs.aoi_gdf.to_crs(epsg=6933).area.sum() / 10000
                except Exception as e:
                    logger.warning(f"Could not calculate AOI area: {e}")

            return SamplingResults(
                sampling_method=self.method,
                success=True,
                total_samples=total_samples,
                target_error=inputs.target_error,
                confidence_level=inputs.confidence_level,
                current_moe_percent=current_moe * 100,
                current_moe_decimal=current_moe,
                precision_curve=precision_curve,
                total_area_ha=total_area_ha,
                allocation_method="systematic",
            )

        except Exception as e:
            logger.error(f"Error in systematic sampling calculation: {e}")
            return SamplingResults.error(self.method, str(e))
