import logging

import solara

from component.model import app_state
from component.scripts.geospatial import generate_sample_points

logger = logging.getLogger("sbae.point_generation")


@solara.component
def PointGeneration(sbae_map):
    """Point generation component for the right panel."""
    is_loading = solara.use_reactive(False)
    use_custom_seed = solara.use_reactive(False)
    custom_seed = solara.use_reactive(42)

    def handle_generate_points():
        """Handle sample point generation."""
        if not app_state.is_ready_for_point_generation():
            app_state.add_error("Please complete sample size calculation first.")
            return

        if is_loading.value:
            return

        is_loading.value = True

        try:
            app_state.set_processing_status("Generating sample points...")

            seed = custom_seed.value if use_custom_seed.value else None

            # Get sampling method from sample results
            sampling_method = "stratified"
            total_samples = None

            if app_state.sample_results.value:
                sampling_method = app_state.sample_results.value.get(
                    "sampling_method", "stratified"
                )
                total_samples = app_state.sample_results.value.get(
                    "total_samples", None
                )

            points_df = generate_sample_points(
                file_path=app_state.file_path.value,
                samples_per_class=app_state.samples_per_class.value,
                class_lookup=app_state.get_class_lookup(),
                seed=seed,
                sampling_method=sampling_method,
                total_samples=total_samples,
            )

            app_state.set_sample_points(points_df)

            logger.debug(f"Generated {len(points_df)} sample points.")

            if sbae_map and points_df is not None and not points_df.empty:
                logger.info("Adding sample points to map...")

                sbae_map.add_sample_points(points_df)

                logger.debug(points_df.head())

            app_state.set_processing_status("")

        except Exception as e:
            app_state.add_error(f"Error generating points: {str(e)}")
            app_state.set_processing_status("")
        finally:
            is_loading.value = False

    # Check if allocation has changed since points were generated
    allocation_changed = False
    if (
        app_state.sample_points.value is not None
        and not app_state.sample_points.value.empty
        and app_state.sample_results.value
    ):
        current_total = app_state.sample_results.value.get("total_samples", 0)
        generated_total = len(app_state.sample_points.value)
        allocation_changed = current_total != generated_total

        # Also check if sampling method has changed
        if not allocation_changed:
            current_method = app_state.sample_results.value.get(
                "sampling_method", "stratified"
            )
            points_method = app_state.points_sampling_method.value

            # If methods don't match, warn user to regenerate
            if points_method and current_method != points_method:
                allocation_changed = True

    sample_results = app_state.sample_results.value
    file_ready = app_state.file_path.value is not None

    # Check if allocation is ready based on sampling method
    allocation_ready = False
    if sample_results:
        sampling_method = sample_results.get("sampling_method", "stratified")
        if sampling_method in ("simple", "systematic"):
            # For non-stratified methods, just need total_samples
            allocation_ready = sample_results.get("total_samples", 0) > 0
        else:
            # For stratified sampling, need samples_per_class
            allocation_ready = bool(app_state.samples_per_class.value)

    with solara.Column():
        if sample_results is None:
            solara.Info("Calculate sample sizes first.")
        else:

            with solara.Row(
                justify="space-between",
                style="align-items: center; margin-bottom: 8px;",
            ):
                solara.Checkbox(
                    label="Use custom seed",
                    value=use_custom_seed.value,
                    on_value=lambda v: setattr(use_custom_seed, "value", v),
                )

                if use_custom_seed.value:
                    solara.v.TextField(
                        label="Seed",
                        v_model=custom_seed.value,
                        on_v_model=lambda v: setattr(
                            custom_seed,
                            "value",
                            int(float(v)) if v and v != "" else 0,
                        ),
                        type="number",
                        step=1,
                        min=0,
                        style="width: 100px;",
                    )

            solara.Button(
                "Generate Points",
                on_click=handle_generate_points,
                color="primary",
                block=True,
                small=True,
                loading=is_loading.value,
            )

            # Warning if allocation changed
            if allocation_changed:
                solara.Warning(
                    "⚠️ Sample allocation has changed! The points shown on the map don't match your current allocation. Please regenerate points."
                )
            if not allocation_ready:
                solara.Info("Sample allocation is missing. Recalculate sample size.")

            if not file_ready:
                solara.Info("Upload a classification map before generating points.")
