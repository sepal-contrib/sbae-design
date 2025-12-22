"""Sample configuration widget using the new sampling architecture.

This module provides the UI for configuring sampling parameters
and delegates calculations to the sampling service.
"""

import logging

import solara

from component.model import app_state
from component.sampling import SamplingService
from component.widget.aoi_upload_selector import AoiUploadSelector

logger = logging.getLogger("sbae.sample_configuration")


@solara.component
def SampleConfiguration(sbae_map=None):
    """Sample configuration widget for the right panel."""
    # Use use_ref to persist value across renders without re-initializing
    prev_method_ref = solara.use_ref(app_state.sampling_method.value)
    current_method = app_state.sampling_method.value

    # Check if method changed - use_ref.current persists between renders
    if prev_method_ref.current != current_method:
        logger.info(
            f">>> METHOD CHANGED from {prev_method_ref.current} to {current_method}, CLEARING ALL DATA"
        )

        # Clean up map layers
        if sbae_map:
            sbae_map.remove_layer("clas", none_ok=True)
            if sbae_map.sample_points_layer:
                try:
                    sbae_map.remove_layer(sbae_map.sample_points_layer)
                except Exception:
                    pass
                sbae_map.sample_points_layer = None

        # Clear all sampling data (both file and AOI)
        app_state.clear_all_sampling_data()

        # Update tracked method AFTER clearing
        prev_method_ref.current = current_method

    def auto_calculate():
        """Auto-calculate when ready and parameters change."""
        if SamplingService.is_ready(app_state):
            run_calculation()

    solara.use_effect(
        auto_calculate,
        [
            app_state.target_error.value,
            app_state.confidence_level.value,
            app_state.min_samples_per_class.value,
            app_state.expected_accuracy.value,
            app_state.sampling_method.value,
            app_state.stratified_allocation_method.value,
            app_state.simple_total_samples.value,
            app_state.area_data.value,
            app_state.aoi_gdf.value,
            app_state.expected_user_accuracies.value,
            app_state.high_eua.value,
            app_state.low_eua.value,
            app_state.eua_modes.value,
        ],
    )

    def run_calculation():
        """Execute the sampling calculation."""
        try:
            results = SamplingService.calculate_from_state(app_state)
            if results.success:
                app_state.set_sample_results(results.to_dict())
            else:
                app_state.add_error(results.error_message or "Calculation failed")
        except Exception as e:
            app_state.add_error(f"Error calculating samples: {str(e)}")

    sampling_method = app_state.sampling_method.value

    with solara.Column():
        SamplingMethodSelector()

        AoiUploadSelector(sbae_map)

        # Check if data source is available for current method
        has_valid_data = False
        if sampling_method in ("simple", "systematic"):
            has_valid_data = app_state.aoi_gdf.value is not None
        elif sampling_method == "stratified":
            has_valid_data = (
                app_state.area_data.value is not None
                and not app_state.area_data.value.empty
            )

        if not has_valid_data:
            return

        if sampling_method in ("simple", "systematic"):
            SimpleSystematicParameters()
        elif sampling_method == "stratified":
            StratifiedParameters()


@solara.component
def SamplingMethodSelector():
    """Dropdown for selecting sampling method."""

    def update_method(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value,
                    app_state.confidence_level.value,
                    app_state.min_samples_per_class.value,
                    app_state.expected_accuracy.value,
                    value,
                    app_state.simple_total_samples.value,
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid sampling method: {str(e)}")

    solara.Select(
        label="Sampling Method",
        value=app_state.sampling_method.value,
        values=["stratified", "simple", "systematic"],
        on_value=update_method,
    )


@solara.component
def SimpleSystematicParameters():
    """Parameters for simple and systematic sampling."""

    def update_total_samples(value):
        if value is not None and value != "":
            try:
                int_value = int(float(value))
                if int_value > 0:
                    app_state.set_sampling_parameters(
                        app_state.target_error.value,
                        app_state.confidence_level.value,
                        app_state.min_samples_per_class.value,
                        app_state.expected_accuracy.value,
                        app_state.sampling_method.value,
                        int_value,
                    )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid sample total: {str(e)}")

    def update_confidence(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value, float(value)
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid confidence level: {str(e)}")

    def update_expected_accuracy(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value,
                    app_state.confidence_level.value,
                    app_state.min_samples_per_class.value,
                    float(value),
                    app_state.sampling_method.value,
                    app_state.simple_total_samples.value,
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid expected accuracy: {str(e)}")

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label="Total Sample Size",
                v_model=app_state.simple_total_samples.value,
                on_v_model=update_total_samples,
                type="number",
            )
        with solara.Column(style="flex: 1;"):
            solara.Select(
                label="Confidence Level",
                value=app_state.confidence_level.value,
                values=[90.0, 95.0, 99.0],
                on_value=update_confidence,
            )

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.SliderFloat(
                "Expected Overall Accuracy (%)",
                value=app_state.expected_accuracy.value,
                min=50.0,
                max=99.0,
                step=1.0,
                on_value=update_expected_accuracy,
            )


@solara.component
def StratifiedParameters():
    """Parameters for stratified sampling."""

    def update_target_error(value):
        if value is not None and value != "":
            try:
                float_value = float(value)
                if float_value > 0:
                    app_state.set_sampling_parameters(
                        float_value, app_state.confidence_level.value
                    )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid target error: {str(e)}")

    def update_min_samples(value):
        if value is not None and value != "":
            try:
                int_value = int(float(value))
                if int_value > 0:
                    app_state.set_sampling_parameters(
                        app_state.target_error.value,
                        app_state.confidence_level.value,
                        int_value,
                    )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid minimum samples: {str(e)}")

    def update_allocation_method(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value,
                    app_state.confidence_level.value,
                    app_state.min_samples_per_class.value,
                    app_state.expected_accuracy.value,
                    app_state.sampling_method.value,
                    app_state.simple_total_samples.value,
                    value,
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid allocation method: {str(e)}")

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label="Target Standard Error (%)",
                v_model=app_state.target_error.value,
                on_v_model=update_target_error,
                type="number",
                hint="Desired precision (e.g., 1% means ±1% margin of error)",
            )

        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label="Minimum Samples per Class",
                v_model=app_state.min_samples_per_class.value,
                on_v_model=update_min_samples,
                type="number",
                hint="Safety minimum for small/rare classes",
            )

    solara.Select(
        label="Allocation Method",
        value=app_state.stratified_allocation_method.value,
        values=["proportional", "equal", "neyman", "balanced"],
        on_value=update_allocation_method,
    )
