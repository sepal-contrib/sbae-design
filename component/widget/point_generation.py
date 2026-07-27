import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import solara

from component.model import app_state
from component.scripts.geospatial import extract_map_codes, generate_sample_points
from component.scripts.vector_tiles import build_layer_or_notify

logger = logging.getLogger("sbae.point_generation")


POINT_GENERATION_RUNNING = "running"
POINT_GENERATION_FINISHED = "finished"
POINT_GENERATION_ERROR = "error"


@dataclass
class PointGenerationController:
    """State and actions for the point-generation UI."""

    custom_seed_enabled: solara.Reactive[bool]
    custom_seed: solara.Reactive[int]
    is_generating: bool
    trigger: Callable[[], None]


def build_point_generation_request(
    state,
    *,
    request_id: int,
    use_custom_seed: bool,
    custom_seed: int,
):
    """Capture all inputs needed by a point-generation worker.

    The worker may outlive the Design tab component that started it, so it must
    not read component-local trigger state after launch.
    """
    sample_results = state.sample_results.value or {}
    sampling_method = sample_results.get("sampling_method", "stratified")

    return {
        "request_id": request_id,
        "seed": custom_seed if use_custom_seed else None,
        "sampling_method": sampling_method,
        "total_samples": sample_results.get("total_samples", None),
        "file_path": state.file_path.value,
        "samples_per_class": dict(state.samples_per_class.value or {}),
        "class_lookup": dict(state.get_class_lookup()),
        "aoi_gdf": state.aoi_gdf.value,
    }


def run_point_generation_request(request):
    """Generate points from a captured request."""
    sampling_method = request["sampling_method"]
    seed = request["seed"]

    logger.debug(
        "Starting point generation request %s with method: %s, seed: %s",
        request["request_id"],
        sampling_method,
        seed,
    )

    if sampling_method in ("simple", "systematic"):
        points_df = generate_sample_points(
            aoi_gdf=request["aoi_gdf"],
            samples_per_class={},
            class_lookup={},
            seed=seed,
            sampling_method=sampling_method,
            total_samples=request["total_samples"],
        )
    else:
        points_df = generate_sample_points(
            file_path=request["file_path"],
            samples_per_class=request["samples_per_class"],
            class_lookup=request["class_lookup"],
            seed=seed,
            sampling_method=sampling_method,
            total_samples=request["total_samples"],
        )

    logger.debug("Generated %s sample points.", len(points_df))
    return points_df


def _result_is_generating(generation_task, generation_request) -> bool:
    return generation_request.value is not None and generation_task.pending


def use_point_generation_task(sbae_map=None) -> PointGenerationController:
    """Own the point-generation thread from a component that survives tab swaps."""
    use_custom_seed = solara.use_reactive(True)
    custom_seed = solara.use_reactive(33)
    generation_request = solara.use_reactive(None)
    request_id_ref = solara.use_ref(0)

    async def generate_points_worker():
        request = generation_request.value
        if request is None:
            return None
        points_df = await asyncio.to_thread(run_point_generation_request, request)
        # Generation leaves map_code=0; sample the design's classification raster
        # at each point to record its real map class (keep all points -- the
        # sample is fixed, so unsampleable ones stay as generated).
        raster = app_state.optimized_raster_path.value or app_state.file_path.value
        if (
            raster
            and points_df is not None
            and not points_df.empty
            and "longitude" in points_df.columns
        ):
            try:
                points_df, _ = await asyncio.to_thread(
                    extract_map_codes,
                    points_df,
                    raster,
                    "longitude",
                    "latitude",
                    drop_missing=False,
                )
            except Exception as e:
                logger.warning("Could not sample map_code from design raster: %s", e)
        layer = await build_layer_or_notify(sbae_map, points_df)
        return (points_df, layer)

    # prefer_threaded defaults to True: this worker is non-GEE (no loop-bound
    # state), and prefer_threaded=False would need a running loop the sync
    # solara.render() tests don't provide -- running in a worker thread is fine.
    generation_task = solara.lab.use_task(
        generate_points_worker,
        dependencies=[generation_request.value],
        raise_error=False,
    )

    def handle_generation_result():
        if generation_request.value is None:
            return

        if generation_task.pending:
            app_state.points_generation_status.value = POINT_GENERATION_RUNNING
            app_state.set_processing_status("Generating sample points...")
        elif generation_task.error:
            app_state.add_error(f"Error generating points: {generation_task.exception}")
            app_state.points_generation_status.value = POINT_GENERATION_ERROR
            app_state.set_processing_status("")
            generation_request.value = None
        elif generation_task.finished:
            result = generation_task.value
            points_df = result[0] if result is not None else None
            layer = result[1] if result is not None else None
            if points_df is not None:
                app_state.set_sample_points(points_df)
                if sbae_map and layer is not None:
                    logger.info("Attaching sample points layer to map...")
                    sbae_map.attach_sample_points_layer(layer)
                app_state.points_generation_status.value = POINT_GENERATION_FINISHED

            app_state.set_processing_status("")
            generation_request.value = None

    solara.use_effect(
        handle_generation_result,
        [generation_task.pending, generation_task.finished, generation_task.error],
    )

    def handle_generate_points():
        """Trigger point generation."""
        if (
            app_state.points_generation_status.value == POINT_GENERATION_RUNNING
            or _result_is_generating(generation_task, generation_request)
        ):
            return

        if not app_state.is_ready_for_point_generation():
            sampling_method = app_state.sampling_method.value
            if sampling_method == "stratified":
                app_state.add_error(
                    "Please upload a classification map and complete sample size calculation first."
                )
            else:
                app_state.add_error(
                    "Please select an Area of Interest and complete sample size calculation first."
                )
            return

        request_id_ref.current += 1
        generation_request.value = build_point_generation_request(
            app_state,
            request_id=request_id_ref.current,
            use_custom_seed=use_custom_seed.value,
            custom_seed=custom_seed.value,
        )
        app_state.points_generation_status.value = POINT_GENERATION_RUNNING
        app_state.set_processing_status("Generating sample points...")

    return PointGenerationController(
        custom_seed_enabled=use_custom_seed,
        custom_seed=custom_seed,
        is_generating=(
            app_state.points_generation_status.value == POINT_GENERATION_RUNNING
            or _result_is_generating(generation_task, generation_request)
        ),
        trigger=handle_generate_points,
    )


@solara.component
def PointGeneration(sbae_map):
    """Point generation component for the right panel."""
    PointGenerationView(sbae_map, use_point_generation_task(sbae_map))


@solara.component
def PointGenerationView(sbae_map, controller: PointGenerationController):
    """Render point-generation controls for an existing task controller."""
    custom_seed_enabled = controller.custom_seed_enabled
    custom_seed = controller.custom_seed
    is_generating = controller.is_generating

    # Get current parameters for dependencies
    sampling_method = (
        app_state.sample_results.value.get("sampling_method", "stratified")
        if app_state.sample_results.value
        else "stratified"
    )

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

    # Check if ready for point generation
    ready_for_generation = app_state.is_ready_for_point_generation()

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
                    value=custom_seed_enabled.value,
                    on_value=lambda v: setattr(custom_seed_enabled, "value", v),
                )

                if custom_seed_enabled.value:
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
                on_click=controller.trigger,
                color="primary",
                block=True,
                small=True,
                loading=is_generating,
                disabled=is_generating,
            )

            # Show generation progress
            if is_generating:
                solara.Info(
                    "Generating sample points... This may take a moment for large datasets."
                )
                solara.ProgressLinear(value=True)

            # Warning if allocation changed
            if allocation_changed:
                solara.Warning(
                    "Sample allocation has changed! The points shown on the map don't match your current allocation. Please regenerate points."
                )
            elif not ready_for_generation:
                sampling_method = sample_results.get("sampling_method", "stratified")
                if sampling_method in ("simple", "systematic"):
                    solara.Info("Select an Area of Interest before generating points.")
                else:
                    solara.Info("Upload a classification map before generating points.")
