import solara
from solara.alias import rv

from component.model import app_state
from component.scripts.calculations import calculate_sample_design


@solara.component
def SampleCalculationTile():
    """Step 3: Calculate Sample Size Dialog."""
    with solara.Column():
        solara.Markdown("## 🔢 Calculate Sample Size")
        solara.Markdown(
            """
            Configure sampling parameters and calculate the required sample size
            for your accuracy assessment based on statistical methods.
            
            **Parameters:**
            - **Target Overall Accuracy**: Desired accuracy level (e.g., 85%)
            - **Confidence Level**: Statistical confidence (typically 95%)
            - **Minimum Sample Size**: Safety minimum for small classes
            """
        )

        sample_size_calculator()

        # Show allocation table if results exist
        if (
            app_state.sample_results.value is not None
            and app_state.sample_results.value
        ):
            sample_allocation_table()
            solara.Success("✅ Sample allocation calculated! Ready to generate points.")


def sample_size_calculator() -> None:
    """Sample size calculation component."""

    def handle_calculate_samples():
        """Handle sample size calculation."""
        if not app_state.is_ready_for_calculation():
            app_state.add_error("Please upload a classification map first.")
            return

        try:
            app_state.set_processing_status("Calculating sample sizes...")

            # Get current parameters
            area_data = app_state.area_data.value
            target_error = (
                app_state.target_error.value / 100.0
            )  # Convert percentage to decimal
            confidence_level = (
                app_state.confidence_level.value / 100.0
            )  # Convert percentage to decimal

            # Calculate sample allocation
            allocation_dict = calculate_sample_design(
                area_df=area_data,
                objective="Overall Accuracy",
                target_oa=0.85,  # Default target
                allowable_error=target_error,
                confidence_level=confidence_level,
                min_samples_per_class=5,  # Default minimum
                allocation_method="Proportional",
            )

            # Create properly formatted results dictionary
            total_samples = sum(allocation_dict.values()) if allocation_dict else 0

            # Create samples per class list with class names
            samples_per_class = []
            for class_code, sample_count in allocation_dict.items():
                # Find class name from area data
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
                "allocation_method": "Proportional",
                "samples_per_class": samples_per_class,
                "allocation_dict": allocation_dict,  # Keep raw allocation for other functions
            }

            app_state.set_sample_results(results)
            app_state.set_processing_status("")

        except Exception as e:
            app_state.add_error(f"Error calculating samples: {str(e)}")
            app_state.set_processing_status("")

    with solara.Card("Calculate Sample Size"):
        solara.Markdown(
            """
        Configure sampling parameters to determine the required sample size
        for your accuracy assessment.
        """
        )

        if app_state.area_data.value.empty:
            solara.Info("Upload and configure your map first.")
            return

        # Parameter update handlers
        def update_target_error(value):
            app_state.set_sampling_parameters(value, app_state.confidence_level.value)

        def update_confidence_level(value):
            app_state.set_sampling_parameters(app_state.target_error.value, value)

        with solara.Row():

            solara.SliderFloat(
                "Target Margin of Error (%)",
                value=app_state.target_error.value,
                min=1.0,
                max=10.0,
                step=0.5,
                on_value=update_target_error,
            )
        with solara.Row():

            solara.SliderFloat(
                "Confidence Level (%)",
                value=app_state.confidence_level.value,
                min=90.0,
                max=99.0,
                step=1.0,
                on_value=update_confidence_level,
            )

        with solara.Row():
            solara.Button(
                "Calculate Sample Size",
                on_click=handle_calculate_samples,
                color="primary",
                outlined=True,
            )

        if app_state.sample_results.value:
            sample_results = app_state.sample_results.value
            with rv.Alert(type="info", text=True):
                solara.Markdown(
                    f"""
                **Sample Size Calculation Results:**
                - Total samples needed: **{sample_results.get("total_samples", 0)}**
                - Allocation method: **{sample_results.get("allocation_method", "Unknown")}**

                **Samples per class:**
                """
                )

                for class_info in sample_results.get("samples_per_class", []):
                    solara.Markdown(
                        f"- {class_info['class_name']}: {class_info['samples']}"
                    )


def sample_allocation_table() -> None:
    """Display sample allocation with manual editing - self-contained with its own logic."""
    if not app_state.sample_results.value:
        return

    allocation_data = app_state.get_allocation_data()

    with solara.Card("📋 Sample Allocation"):
        if not allocation_data:
            solara.Warning("No allocation data available")
            return

        solara.Markdown("**Manual Allocation Editing:**")

        for item in allocation_data:
            with solara.Columns([6, 6]):
                solara.Text(f"{item['class_name']}:")

                def make_update_callback(code):
                    def update_samples(samples):
                        app_state.update_manual_allocation(code, int(samples))

                    return update_samples

                solara.InputInt(
                    label="Samples",
                    value=item["samples"],
                    on_value=make_update_callback(item["map_code"]),
                )
