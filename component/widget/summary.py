from typing import Dict, Optional

import pandas as pd
import solara

from component.model import app_state


@solara.component
def Summary():
    """Right panel content with progress and summary."""
    with solara.Column():
        # Statistics summary
        statistics_summary(
            area_data=app_state.area_data.value,
            sample_results=app_state.sample_results.value,
            sample_points=app_state.sample_points.value,
        )


def statistics_summary(
    area_data: Optional[pd.DataFrame] = None,
    sample_results: Optional[Dict] = None,
    sample_points: Optional[pd.DataFrame] = None,
) -> None:
    """Display summary statistics.

    Args:
        area_data: DataFrame with area information
        sample_results: Sample calculation results
        sample_points: Generated sample points
    """
    with solara.Card("Project Summary"):
        if area_data is not None and not area_data.empty:
            total_area = area_data["map_area"].sum() / 10000  # Convert to hectares
            n_classes = len(area_data)

            solara.Markdown(
                f"""
            **Map Statistics:**
            - Total area: {total_area:,.1f} hectares
            - Number of classes: {n_classes}
            """
            )

        if sample_results:
            solara.Markdown(
                f"""
            **Sampling Design:**
            - Target error: {sample_results.get("target_error", "N/A")}%
            - Confidence level: {sample_results.get("confidence_level", "N/A")}%
            - Total samples: {sample_results.get("total_samples", "N/A")}
            """
            )

        if sample_points is not None and not sample_points.empty:
            points_per_class = sample_points.groupby("map_code").size()

            solara.Markdown(
                f"""
            **Generated Points:**
            - Total points: {len(sample_points)}
            - Classes sampled: {len(points_per_class)}
            """
            )
