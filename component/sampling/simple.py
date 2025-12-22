"""Simple random sampling strategy implementation.

Simple random sampling gives every point in the AOI an equal probability
of being selected. This is the most basic sampling method and serves as
the baseline for comparison with other methods.
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

logger = logging.getLogger("sbae.sampling.simple")


class SimpleSamplingStrategy(SamplingStrategy):
    """Strategy for simple random sampling.

    Simple random sampling is ideal when:
    - You don't have a classification map
    - You want to estimate a single overall accuracy value
    - All areas have equal importance
    """

    @property
    def method(self) -> SamplingMethod:
        return SamplingMethod.SIMPLE

    @property
    def display_name(self) -> str:
        return "Simple Random Sampling"

    @property
    def description(self) -> str:
        return (
            "Randomly distribute sample points across the entire AOI with equal "
            "probability. Best for general accuracy assessment without class stratification."
        )

    @property
    def requires_aoi(self) -> bool:
        return True

    @property
    def supports_precision_curve(self) -> bool:
        return True

    def validate_inputs(self, inputs: SamplingInputs) -> List[str]:
        """Validate inputs for simple random sampling."""
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
        """Calculate simple random sample design."""
        # Validate first
        errors = self.validate_inputs(inputs)
        if errors:
            return SamplingResults.error(self.method, "; ".join(errors))

        try:
            total_samples = inputs.total_samples
            target_oa = inputs.expected_accuracy_decimal
            confidence_level = inputs.confidence_level_decimal

            # Calculate current MOE for the specified sample size
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

            # Calculate AOI area using appropriate projection
            total_area_ha = 0.0
            if inputs.aoi_gdf is not None:
                try:
                    gdf = inputs.aoi_gdf
                    if gdf.crs and gdf.crs.is_geographic:
                        # Use UTM zone appropriate for this AOI's location
                        gdf = gdf.to_crs(gdf.estimate_utm_crs())
                    total_area_ha = gdf.geometry.area.sum() / 10000
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
                allocation_method="simple",
            )

        except Exception as e:
            logger.error(f"Error in simple random sampling calculation: {e}")
            return SamplingResults.error(self.method, str(e))
