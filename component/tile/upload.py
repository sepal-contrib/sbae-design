import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import solara
from pysepal.mapping import prepare_for_tiles
from pysepal.solara import use_notifications
from pysepal.solara.components.inputs import FileInputComponent

from component.message import get_translator, use_translator
from component.model import app_state
from component.scripts.geospatial import (
    compute_file_areas,
    get_color_palette,
    get_file_info,
    is_raster_file,
)
from component.widget.map import SbaeMap

logger = logging.getLogger("sbae.upload")


@solara.component
def RasterMapWatcher(sbae_map: SbaeMap):
    """Watches for optimized raster and adds it to map. Must stay mounted."""
    ms = use_translator()

    def add_optimized_raster_to_map():
        optimized_path = app_state.optimized_raster_path.value
        status = app_state.raster_optimization_status.value
        sampling_method = app_state.sampling_method.value
        class_colors = app_state.class_colors.value

        # The palette and the tiled COG are produced by two independent async
        # paths, so wait for the palette: adding the raster without it renders
        # every class down the dark end of the default continuous ramp. Holding
        # the status here is what re-runs this effect once the palette lands.
        if (
            optimized_path
            and status == "adding_to_map"
            and sampling_method == "stratified"
            and class_colors
        ):
            sbae_map.add_raster(
                optimized_path,
                layer_name=ms.upload.layer_name,
                key="clas",
                class_colors=class_colors,
            )
            app_state.raster_optimization_status.value = "finished"

    solara.use_effect(
        add_optimized_raster_to_map,
        [
            app_state.optimized_raster_path.value,
            app_state.raster_optimization_status.value,
            app_state.sampling_method.value,
            app_state.class_colors.value,
        ],
    )


@solara.component
def CurrentFileDisplay(sbae_map: SbaeMap = None):
    """Display the currently selected file with option to clear it."""
    ms = use_translator()

    def clear_file():
        """Clear the current file and reset related state."""
        # Remove map layers first
        if sbae_map:
            sbae_map.remove_layer("clas", none_ok=True)
            if sbae_map.sample_points_layer:
                try:
                    sbae_map.remove_layer(sbae_map.sample_points_layer)
                except Exception:
                    pass
                sbae_map.sample_points_layer = None

        # Clear all file-related state
        app_state.clear_file_data()

        # Reset workflow step
        if app_state.current_step.value > 1:
            app_state.current_step.value = 1

    if app_state.uploaded_file_info.value is None or app_state.file_path.value is None:
        return

    file_info = app_state.uploaded_file_info.value
    file_path = app_state.file_path.value
    file_name = Path(file_path).name
    optimization_status = app_state.raster_optimization_status.value
    is_loading = optimization_status in ("running", "adding_to_map")

    with solara.Card(classes=["mb-4"]):
        with solara.Row(justify="space-between", style={"align-items": "center"}):
            with solara.Column(gap="0px"):
                solara.HTML(
                    tag="div",
                    unsafe_innerHTML=(
                        f"<strong>{ms.upload.current_file}</strong> {file_name}"
                    ),
                    style="font-size: 14px;",
                )
                file_type = file_info.get("file_type", "unknown").title()
                size_mb = file_info.get("size_mb", 0)
                solara.HTML(
                    tag="div",
                    unsafe_innerHTML=ms.upload.current_file_details.format(
                        file_type, f"{size_mb:.1f}"
                    ),
                    style="font-size: 12px; margin-top: 4px;",
                )

            solara.Button(
                label="",
                icon_name="mdi-close",
                on_click=clear_file,
                color="error",
                text=True,
                icon=True,
            )
        (
            solara.v.ProgressLinear(indeterminate=is_loading, classes=["my-2"])
            if is_loading
            else None
        )


def _reject_reason(file_info: dict, ms=None) -> Optional[str]:
    """Why this file cannot serve as the classification map, or ``None``.

    Raster only: the map is served as tiles and the stratified design reads its
    classes per pixel, so a vector carries neither. ``get_file_info`` reports
    ``"vector"`` for one and ``"unknown"`` for anything it could not open.
    """
    ms = ms if ms is not None else get_translator()
    if "error" in file_info:
        return file_info["error"]
    if file_info.get("file_type") != "raster":
        return ms.upload.error.not_a_raster
    return None


def _upload_toast(*, has_file, is_raster, state, value, error, ms=None):
    """Decide the terminal upload toast: ``(level, message)`` or ``None``.

    The raster success toast fires only when the optimization produced a real
    result (``value``), never on the idle/no-op ``FINISHED`` state of the
    prep thread's ``lambda: None`` branch. That ambiguity -- the thread rests at
    ``FINISHED`` before the real run starts -- is what fired the "optimized"
    toast twice.
    """
    ms = ms if ms is not None else get_translator()
    if not has_file:
        return None
    if not is_raster:
        return ("success", ms.upload.toast.uploaded)
    if state == solara.ResultState.ERROR:
        return ("error", ms.upload.toast.optimization_failed.format(error))
    if state == solara.ResultState.FINISHED and value:
        return ("success", ms.upload.toast.optimized)
    return None


@solara.component
def UploadTile(sbae_map: SbaeMap):
    """Step 1: File Upload Dialog."""
    ms = use_translator()
    is_loading = solara.use_reactive(False)

    has_file = (
        app_state.uploaded_file_info.value is not None
        and app_state.file_path.value is not None
    )

    def prepare_raster_worker(file_path):
        """Worker function for raster tiling in separate thread."""

        def worker():
            original_file_path = file_path
            app_state.raster_optimization_status.value = "running"
            app_state.raster_optimization_error.value = None
            app_state.optimized_raster_path.value = None
            try:
                prep = prepare_for_tiles(file_path, warp_to_3857=True)

                # Check if file was cleared/changed during processing
                if app_state.file_path.value != original_file_path:
                    logger.debug("File changed during optimization, discarding result")
                    return None

                app_state.optimized_raster_path.value = prep["path"]
                app_state.raster_optimization_status.value = "adding_to_map"
                return prep
            except Exception as e:
                # Only set error if file is still the same
                if app_state.file_path.value == original_file_path:
                    app_state.raster_optimization_status.value = "error"
                    app_state.raster_optimization_error.value = str(e)
                raise

        return worker

    # Thread for raster preparation - only starts if no optimized path exists
    raster_prep_result = solara.use_thread(
        (
            prepare_raster_worker(app_state.file_path.value)
            if (
                has_file
                and is_raster_file(app_state.file_path.value or "")
                and not app_state.optimized_raster_path.value
            )
            else lambda: None
        ),
        dependencies=[app_state.file_path.value],
        intrusive_cancel=False,
    )

    def handle_layer_removal():
        """Take the classification layer off the map when it no longer applies.

        Adding it is ``RasterMapWatcher``'s job, once the tiled COG and the
        class palette are both ready.
        """
        if sbae_map is None:
            return
        sampling_method = app_state.sampling_method.value
        should_show_layer = has_file and sampling_method == "stratified"

        if not should_show_layer:
            sbae_map.remove_layer("clas", none_ok=True)

    solara.use_effect(
        handle_layer_removal,
        [
            has_file,
            app_state.file_path.value,
            app_state.sampling_method.value,
        ],
    )

    notifications = use_notifications()

    def announce_upload_state():
        """Toast the terminal upload/optimization state (progress stays inline)."""
        toast = _upload_toast(
            has_file=has_file,
            is_raster=is_raster_file(app_state.file_path.value or ""),
            state=raster_prep_result.state,
            value=raster_prep_result.value,
            error=raster_prep_result.error,
            ms=ms,
        )
        if toast is None:
            return
        level, message = toast
        if level == "error":
            logger.error("Raster optimization failed: %s", raster_prep_result.error)
            notifications.error(message)
        else:
            notifications.success(message)

    solara.use_effect(announce_upload_state, [has_file, raster_prep_result.state])

    with solara.Column():
        FileUploadSection(is_loading=is_loading)

        # In-modal progress for raster optimization stays inline (a toast would
        # auto-dismiss mid-operation); the success/error toast fires on finish.
        if (
            has_file
            and is_raster_file(app_state.file_path.value or "")
            and raster_prep_result.state == solara.ResultState.RUNNING
        ):
            solara.Text(
                ms.upload.optimizing,
                style="font-size: 13px; opacity: 0.8;",
            )
            solara.ProgressLinear(value=True)


@solara.component
def SampleMapButton(is_loading: solara.Reactive[bool]):
    """Button to load sample map for testing."""
    ms = use_translator()

    def load_sample_map():
        """Load the sample map for testing."""
        sample_file_path = (
            Path(__file__).parent.parent.parent / "tests/data" / "aa_test_congo.tif"
        )

        if is_loading.value:  # Prevent multiple simultaneous loads
            return

        is_loading.value = True
        app_state.error_messages.value = []  # Clear errors directly
        app_state.processing_status.value = ms.upload.loading_sample

        try:
            # Check if file exists
            if not os.path.exists(sample_file_path):
                app_state.error_messages.value = [
                    ms.upload.error.sample_not_found.format(sample_file_path)
                ]
                return

            # Get file information and compute areas
            file_info = get_file_info(sample_file_path)
            area_data = compute_file_areas(sample_file_path)

            # Extract color palette from file
            class_codes = area_data["map_code"].tolist()
            color_palette = get_color_palette(sample_file_path, class_codes)

            # Initialize EUA values for all classes (default to 'high' mode)
            eua_dict = {}
            eua_modes_dict = {}
            for code in class_codes:
                eua_dict[code] = app_state.high_eua.value  # Default to high EUA
                eua_modes_dict[code] = "high"  # Default mode

            # Update state directly
            app_state.uploaded_file_info.value = file_info
            app_state.file_path.value = sample_file_path
            app_state.area_data.value = area_data.copy()
            app_state.original_area_data.value = area_data.copy()
            app_state.class_colors.value = color_palette
            app_state.expected_user_accuracies.value = eua_dict
            app_state.eua_modes.value = eua_modes_dict
            app_state.current_step.value = max(app_state.current_step.value, 2)

        except Exception as e:
            app_state.error_messages.value = [ms.upload.error.sample_failed.format(e)]
        finally:
            app_state.processing_status.value = ""
            is_loading.value = False

    solara.Button(
        ms.upload.sample_map,
        on_click=load_sample_map,
        color="default",
        text=True,
        small=True,
        loading=is_loading.value,
    )


@solara.component
def FileUploadSection(is_loading: solara.Reactive[bool]):
    """File upload component for classification maps."""
    ms = use_translator()
    selected_file_path = solara.use_reactive(None)
    selected_file_info_preview = solara.use_reactive(None)
    is_valid_file = solara.use_reactive(False)
    should_compute_areas = solara.use_reactive(False)

    def reset_all_state():
        """Reset all application state including map."""
        app_state.clear_file_data()
        app_state.processing_status.value = ""
        selected_file_path.value = None
        selected_file_info_preview.value = None
        is_valid_file.value = False
        should_compute_areas.value = False

    def handle_file_selection_from_input(file_path):
        """Handle file selection from FileInputComponent (returns path directly)."""
        if not file_path:
            reset_all_state()
            return

        app_state.file_error.value = None

        try:
            file_info_dict = get_file_info(file_path)

            rejection = _reject_reason(file_info_dict, ms)
            if rejection:
                app_state.file_error.value = rejection
                selected_file_path.value = None
                selected_file_info_preview.value = None
                is_valid_file.value = False
                return

            selected_file_path.value = file_path
            selected_file_info_preview.value = file_info_dict
            is_valid_file.value = True
            app_state.file_error.value = None

        except Exception as e:
            app_state.file_error.value = str(e)
            selected_file_path.value = None
            selected_file_info_preview.value = None
            is_valid_file.value = False

    def compute_areas_worker():
        """Worker function for area computation in separate thread."""
        if not selected_file_path.value or not should_compute_areas.value:
            return None
        area_data = compute_file_areas(selected_file_path.value)
        class_codes = area_data["map_code"].tolist()
        color_palette = get_color_palette(selected_file_path.value, class_codes)
        return {"area_data": area_data, "color_palette": color_palette}

    # Use thread for area computation
    area_result = solara.use_thread(
        compute_areas_worker,
        dependencies=[selected_file_path.value, should_compute_areas.value],
        intrusive_cancel=False,
    )

    # Handle area computation result
    def handle_area_result():
        if area_result.state == solara.ResultState.RUNNING:
            is_loading.value = True
            app_state.processing_status.value = ms.upload.computing_areas
        elif area_result.state == solara.ResultState.ERROR:
            app_state.file_error.value = str(area_result.error)
            app_state.processing_status.value = ""
            is_loading.value = False
            should_compute_areas.value = False
        elif (
            area_result.state == solara.ResultState.FINISHED
            and area_result.value
            and should_compute_areas.value
        ):
            result = area_result.value

            # Initialize EUA values for all classes (default to 'high' mode)
            class_codes = result["area_data"]["map_code"].tolist()
            eua_dict = {}
            eua_modes_dict = {}
            for code in class_codes:
                eua_dict[code] = app_state.high_eua.value  # Default to high EUA
                eua_modes_dict[code] = "high"  # Default mode

            app_state.uploaded_file_info.value = selected_file_info_preview.value
            app_state.file_path.value = selected_file_path.value
            app_state.area_data.value = result["area_data"].copy()
            app_state.original_area_data.value = result["area_data"].copy()
            app_state.class_colors.value = result["color_palette"]
            app_state.expected_user_accuracies.value = eua_dict
            app_state.eua_modes.value = eua_modes_dict
            app_state.current_step.value = max(app_state.current_step.value, 2)
            app_state.file_error.value = None
            app_state.processing_status.value = ""
            is_loading.value = False
            should_compute_areas.value = False
        elif area_result.state == solara.ResultState.FINISHED and not area_result.value:
            is_loading.value = False

    solara.use_effect(handle_area_result, [area_result.state])

    def confirm_file_upload():
        """Trigger area computation for the selected file."""
        if is_loading.value or not selected_file_path.value or not is_valid_file.value:
            return
        is_loading.value = True
        should_compute_areas.value = True

    # Check if currently processing
    is_processing = (
        area_result.state == solara.ResultState.RUNNING
        or area_result.state == solara.ResultState.STARTING
        or is_loading.value
    )

    notifications = use_notifications()

    def announce_file_error():
        if app_state.file_error.value:
            logger.warning("File selection error: %s", app_state.file_error.value)
            notifications.error(app_state.file_error.value)

    solara.use_effect(announce_file_error, [app_state.file_error.value])

    # No card wrapper here: this section renders inside the upload dialog's
    # CardText, so a card of its own would nest cards. See UploadDialogCard.
    with solara.Column(gap="8px"):
        FileUploadInstructions()
        FileInputComponent(on_value=handle_file_selection_from_input)

        if selected_file_info_preview.value and not app_state.uploaded_file_info.value:
            FilePreview(selected_file_info_preview.value)

        if not app_state.uploaded_file_info.value:
            solara.Button(
                ms.upload.confirm,
                on_click=confirm_file_upload,
                color="primary",
                block=True,
                loading=is_processing,
                disabled=not is_valid_file.value or is_processing,
            )

        with solara.Row(justify="center", classes=["mt-4"]):
            solara.Text(ms.common.or_divider)
            SampleMapButton(is_loading=is_loading)


@solara.component
def FileUploadInstructions():
    """Instructions for file upload formats."""
    ms = use_translator()
    solara.Text(ms.upload.instructions)


@solara.component
def FilePreview(file_info: Dict[str, Any]):
    """Preview of the selected file's details, shown before confirmation.

    A neutral, theme-aware panel (subtle border, no colored alert background).
    """
    ms = use_translator()
    rows = [
        (ms.upload.preview.type, file_info.get("file_type", "unknown").title()),
        (ms.upload.preview.size, f"{file_info.get('size_mb', 0):.1f} MB"),
        (ms.upload.preview.features, f"{file_info.get('feature_count', 0):,}"),
        (ms.upload.preview.crs, file_info.get("crs", ms.upload.preview.crs_missing)),
    ]
    with solara.Column(
        gap="2px",
        style=(
            "padding: 10px 12px; border-radius: 6px; "
            "border: 1px solid var(--v-divider-base, rgba(0, 0, 0, 0.12));"
        ),
    ):
        solara.Text(
            ms.upload.preview.title, style="font-weight: 600; margin-bottom: 4px;"
        )
        for label, value in rows:
            with solara.Row(gap="8px"):
                solara.Text(f"{label}:", style="min-width: 72px;")
                solara.Text(str(value))
