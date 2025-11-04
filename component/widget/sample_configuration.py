import logging

import solara

from component.model import app_state
from component.scripts.calculations import calculate_sample_design
from component.scripts.precision import calculate_current_moe, calculate_precision_curve
from component.scripts.simple_random import calculate_overall_accuracy_sample_size

logger = logging.getLogger("sbae.sample_configuration")


@solara.component
def SampleConfiguration(sbae_map=None):
    """Sample configuration widget for the right panel."""
    # Track previous parameters to detect changes
    prev_target_error = solara.use_reactive(app_state.target_error.value)
    prev_confidence_level = solara.use_reactive(app_state.confidence_level.value)
    prev_sampling_method = solara.use_reactive(app_state.sampling_method.value)

    def reset_sample_results():
        """Reset sample results when sampling method changes."""
        logger.debug("Resetting sample results due to sampling method change")
        if prev_sampling_method.value != app_state.sampling_method.value:
            app_state.set_sample_results(None)
            app_state.set_sample_points(None)
            if sbae_map and sbae_map.sample_points_layer:
                sbae_map.map.remove_layer(sbae_map.sample_points_layer)
                sbae_map.sample_points_layer = None
            prev_sampling_method.value = app_state.sampling_method.value

    solara.use_effect(reset_sample_results, [app_state.sampling_method.value])

    def auto_calculate():
        """Auto-calculate samples when parameters change."""
        if app_state.is_ready_for_calculation():
            handle_calculate_samples()

    # Auto-calculate when any parameter changes
    solara.use_effect(
        auto_calculate,
        [
            app_state.target_error.value,
            app_state.min_samples_per_class.value,
            app_state.sampling_method.value,
            app_state.stratified_allocation_method.value,
            app_state.simple_total_samples.value,
            app_state.area_data.value,
            app_state.expected_user_accuracies.value,
            app_state.high_eua.value,
            app_state.low_eua.value,
            app_state.eua_modes.value,
            # These are only used for simple/systematic sampling:
            app_state.confidence_level.value,
            app_state.expected_accuracy.value,
        ],
    )

    def handle_calculate_samples():
        """Handle sample size calculation."""
        if not app_state.is_ready_for_calculation():
            return

        try:
            # Check if parameters have changed
            params_changed = (
                prev_target_error.value != app_state.target_error.value
                or prev_confidence_level.value != app_state.confidence_level.value
            )

            # Clear existing sample points only if parameters changed
            if params_changed:
                app_state.set_sample_points(None)
                if sbae_map and sbae_map.sample_points_layer:
                    sbae_map.map.remove_layer(sbae_map.sample_points_layer)
                    sbae_map.sample_points_layer = None

            # Update previous values
            prev_target_error.value = app_state.target_error.value
            prev_confidence_level.value = app_state.confidence_level.value

            area_data = app_state.area_data.value
            target_error = app_state.target_error.value / 100.0
            confidence_level = app_state.confidence_level.value / 100.0
            min_samples = app_state.min_samples_per_class.value
            expected_oa = app_state.expected_accuracy.value / 100.0
            sampling_method = app_state.sampling_method.value
            allocation_method = (
                app_state.stratified_allocation_method.value.capitalize()
            )

            # Get per-class expected accuracies
            expected_accuracies = app_state.expected_user_accuracies.value

            # Determine total override for simple/systematic sampling
            total_override = None
            if sampling_method in ("simple", "systematic"):
                total_override = int(app_state.simple_total_samples.value)
                allocation_method = sampling_method.capitalize()

            allocation_dict = calculate_sample_design(
                area_df=area_data,
                objective="Overall Accuracy",
                target_oa=expected_oa,
                allowable_error=target_error,
                confidence_level=confidence_level,
                min_samples_per_class=min_samples,
                allocation_method=allocation_method,
                total_samples_override=total_override,
                expected_accuracies=expected_accuracies,
            )

            total_samples = (
                sum(allocation_dict.values())
                if allocation_dict
                else total_override
                if total_override
                else 0
            )

            precision_curve_df = calculate_precision_curve(
                target_oa=expected_oa,
                confidence_level=confidence_level,
                min_sample_size=30,
                max_sample_size=max(1000, int(total_samples * 2)),
                num_points=50,
            )

            # Calculate current MOE for the calculated sample size
            current_moe = calculate_current_moe(
                current_sample_size=total_samples,
                target_oa=expected_oa,
                confidence_level=confidence_level,
            )

            samples_per_class = []
            # For stratified sampling, build per-class allocation
            if allocation_dict:
                for class_code, sample_count in allocation_dict.items():
                    class_row = area_data[area_data["map_code"] == class_code]
                    class_name = (
                        class_row["map_edited_class"].iloc[0]
                        if not class_row.empty
                        else f"Class {class_code}"
                    )

                    samples_per_class.append(
                        {
                            "map_code": class_code,
                            "class_name": class_name,
                            "samples": int(sample_count),
                        }
                    )

            results = {
                "target_error": app_state.target_error.value,
                "confidence_level": app_state.confidence_level.value,
                "total_samples": total_samples,
                "allocation_method": allocation_method,
                "samples_per_class": samples_per_class,
                "allocation_dict": allocation_dict,
                "precision_curve": precision_curve_df.to_dict("records"),
                "current_moe_percent": current_moe * 100,
                "current_moe_decimal": current_moe,
                "sampling_method": sampling_method,
            }

            app_state.set_sample_results(results)

        except Exception as e:
            app_state.add_error(f"Error calculating samples: {str(e)}")

    area_data = app_state.area_data.value

    with solara.Column():
        if area_data is None or area_data.empty:
            if area_data is None:
                logger.debug("Area data is None")
            else:
                logger.debug("Area data is empty")
                logger.debug(area_data)
            solara.Info("Upload a classification map first.")
            return

        sampling_method = app_state.sampling_method.value

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

        def update_confidence_level(value):
            if value is not None:
                try:
                    float_value = float(value)
                    if float_value > 0:
                        app_state.set_sampling_parameters(
                            app_state.target_error.value, float_value
                        )
                except (ValueError, TypeError) as e:
                    app_state.add_error(f"Invalid confidence level: {str(e)}")

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

        def update_expected_accuracy(value):
            if value is not None and value != "":
                try:
                    float_value = float(value)
                    if float_value > 0:
                        app_state.set_sampling_parameters(
                            app_state.target_error.value,
                            app_state.confidence_level.value,
                            app_state.min_samples_per_class.value,
                            float_value,
                            app_state.sampling_method.value,
                            app_state.simple_total_samples.value,
                        )
                except (ValueError, TypeError) as e:
                    app_state.add_error(f"Invalid expected accuracy: {str(e)}")

        def update_sampling_method(value):
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

        def update_simple_total_samples(value):
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
                    app_state.add_error(f"Invalid simple sample total: {str(e)}")

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

        target_error = app_state.target_error.value / 100.0
        confidence_level = app_state.confidence_level.value / 100.0
        expected_oa = app_state.expected_accuracy.value / 100.0

        if sampling_method == "stratified":
            try:
                if target_error <= 0:
                    solara.Error("Target margin of error must be greater than 0%")
                else:
                    # Use stratified formula if we have EUA values
                    if app_state.expected_user_accuracies.value:
                        from component.scripts.calculations import (
                            calculate_stratified_sample_size,
                        )

                        calculate_stratified_sample_size(
                            area_df=area_data,
                            expected_accuracies=app_state.expected_user_accuracies.value,
                            target_standard_error=target_error,
                        )
                    else:
                        # Fallback to simple formula
                        calculate_overall_accuracy_sample_size(
                            target_oa=expected_oa,
                            allowable_error=target_error,
                            confidence_level=confidence_level,
                        )
            except ValueError as e:
                solara.Error(f"Invalid parameters: {str(e)}")

            try:
                calculate_sample_design(
                    area_df=area_data,
                    objective="Overall Accuracy",
                    target_oa=expected_oa,
                    allowable_error=target_error,
                    confidence_level=confidence_level,
                    min_samples_per_class=app_state.min_samples_per_class.value,
                    allocation_method=app_state.stratified_allocation_method.value.capitalize(),
                    expected_accuracies=app_state.expected_user_accuracies.value,
                )
            except Exception as e:
                {}
                solara.Error(f"Failed to calculate allocation: {str(e)}")

        elif sampling_method == "simple":
            app_state.simple_total_samples.value

        solara.Select(
            label="Sampling Method",
            value=app_state.sampling_method.value,
            values=["stratified", "simple", "systematic"],
            on_value=update_sampling_method,
        )

        if sampling_method == "simple":
            with solara.Row(gap="8px", style="margin-bottom: 8px;"):
                with solara.Column(style="flex: 1;"):
                    solara.v.TextField(
                        label="Total Sample Size",
                        v_model=app_state.simple_total_samples.value,
                        on_v_model=update_simple_total_samples,
                        type="number",
                    )
                with solara.Column(style="flex: 1;"):
                    solara.Select(
                        label="Confidence Level",
                        value=app_state.confidence_level.value,
                        values=[90.0, 95.0, 99.0],
                        on_value=update_confidence_level,
                    )

        elif sampling_method == "systematic":
            with solara.Row(gap="8px", style="margin-bottom: 8px;"):
                with solara.Column(style="flex: 1;"):
                    solara.v.TextField(
                        label="Total Sample Size",
                        v_model=app_state.simple_total_samples.value,
                        on_v_model=update_simple_total_samples,
                        type="number",
                    )
                with solara.Column(style="flex: 1;"):
                    solara.Select(
                        label="Confidence Level",
                        value=app_state.confidence_level.value,
                        values=[90.0, 95.0, 99.0],
                        on_value=update_confidence_level,
                    )

        elif sampling_method == "stratified":
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
                values=["proportional", "equal", "neyman"],
                on_value=update_allocation_method,
            )
