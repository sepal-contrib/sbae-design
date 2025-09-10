import logging
import os
from typing import Any, Dict

import solara
from solara.alias import rv

from component.model import app_state
from component.scripts.geospatial import (
    compute_file_areas,
    get_file_info,
    save_uploaded_file,
)
from component.widget.map import SbaeMap

logger = logging.getLogger("sbae.upload")


@solara.component
def UploadTile(sbae_map: SbaeMap):
    """Step 1: File Upload Dialog."""
    # Local state for upload process
    is_loading = solara.use_reactive(False)

    # Derived state - check if file is uploaded
    has_file = (
        app_state.uploaded_file_info.value is not None
        and app_state.file_path.value is not None
    )

    # Effect: Add raster to map when file is uploaded
    def add_to_map():
        if has_file:
            file_path = app_state.file_path.value
            uploaded_file_info = app_state.uploaded_file_info.value
            logger.debug(
                "Adding uploaded classification layer to map. File path: {}, Info: {}",
                file_path,
                uploaded_file_info,
            )
            sbae_map.map.add_raster(file_path)

    # Use effect to handle side effects
    solara.use_memo(add_to_map, [has_file, app_state.file_path.value])

    with solara.Column():
        UploadInstructions()
        SampleMapButton(is_loading=is_loading)
        FileUploadSection(is_loading=is_loading)

        # Declarative success message
        if has_file:
            solara.Success(
                "✅ File uploaded successfully! You can now proceed to edit class names."
            )


@solara.component
def UploadInstructions():
    """Instructions for file upload."""
    solara.Markdown(
        """
        Upload your land cover classification map to begin the sampling process.
        Supported formats include GeoTIFF, Shapefile, GeoJSON, and GeoPackage.
        
        The map will be displayed in the background and used to generate sample points.
        """
    )


@solara.component
def SampleMapButton(is_loading: solara.Reactive[bool]):
    """Button to load sample map for testing."""

    def load_sample_map():
        """Load the sample map for testing."""
        sample_file_path = "/home/dguerrero/Downloads/aa_test_congo.tif"

        if is_loading.value:  # Prevent multiple simultaneous loads
            return

        is_loading.value = True
        app_state.error_messages.value = []  # Clear errors directly
        app_state.processing_status.value = "Loading sample map..."

        try:
            # Check if file exists
            if not os.path.exists(sample_file_path):
                app_state.error_messages.value = [
                    f"Sample file not found: {sample_file_path}"
                ]
                return

            # Get file information and compute areas
            file_info = get_file_info(sample_file_path)
            area_data = compute_file_areas(sample_file_path)

            # Update state directly
            app_state.uploaded_file_info.value = file_info
            app_state.file_path.value = sample_file_path
            app_state.area_data.value = area_data.copy()
            app_state.original_area_data.value = area_data.copy()
            app_state.current_step.value = max(app_state.current_step.value, 2)

        except Exception as e:
            app_state.error_messages.value = [f"Error loading sample map: {str(e)}"]
        finally:
            app_state.processing_status.value = ""
            is_loading.value = False

    solara.Button(
        "Upload Sample Map",
        on_click=load_sample_map,
        color="success",
        outlined=True,
        block=True,
        loading=is_loading.value,
    )


@solara.component
def FileUploadSection(is_loading: solara.Reactive[bool]):
    """File upload component for classification maps."""

    def handle_file_upload(file_info):
        """Handle file upload and processing."""
        if is_loading.value:  # Prevent multiple simultaneous uploads
            return

        is_loading.value = True
        app_state.error_messages.value = []  # Clear errors directly
        app_state.processing_status.value = "Processing uploaded file..."

        try:
            # Save uploaded file
            file_path = save_uploaded_file(file_info, app_state.temp_dir.value)

            # Get file information
            file_info_dict = get_file_info(file_path)

            if "error" in file_info_dict:
                app_state.file_error.value = file_info_dict["error"]
                return

            # Compute areas
            area_data = compute_file_areas(file_path)

            # Update state directly
            app_state.uploaded_file_info.value = file_info_dict
            app_state.file_path.value = file_path
            app_state.area_data.value = area_data.copy()
            app_state.original_area_data.value = area_data.copy()
            app_state.current_step.value = max(app_state.current_step.value, 2)
            app_state.file_error.value = None

        except Exception as e:
            app_state.file_error.value = str(e)
        finally:
            app_state.processing_status.value = ""
            is_loading.value = False

    with solara.Card():
        FileUploadInstructions()

        solara.FileDrop(
            on_file=handle_file_upload,
            lazy=False,
            label="Drop your classification file here or click to browse",
            # disabled=is_loading.value,
        )

        # Declarative error display
        if app_state.file_error.value:
            ErrorAlert(app_state.file_error.value)

        # Declarative success display
        if app_state.uploaded_file_info.value:
            SuccessAlert(app_state.uploaded_file_info.value)


@solara.component
def FileUploadInstructions():
    """Instructions for file upload formats."""
    solara.Markdown(
        """
    Upload your land cover classification map in one of these formats:
    - **Raster**: GeoTIFF (.tif), ERDAS Imagine (.img)
    - **Vector**: Shapefile (.shp), GeoJSON (.geojson), GeoPackage (.gpkg)
    """
    )


@solara.component
def ErrorAlert(error_message: str):
    """Error alert component."""
    with rv.Alert(type="error", text=True):
        solara.Markdown(f"**Error:** {error_message}")


@solara.component
def SuccessAlert(file_info: Dict[str, Any]):
    """Success alert component showing file information."""
    with rv.Alert(type="success", text=True):
        solara.Markdown(
            f"""
        **File uploaded successfully!**
        - Type: {file_info.get("file_type", "unknown").title()}
        - Size: {file_info.get("size_mb", 0):.1f} MB
        - Features: {file_info.get("feature_count", 0):,}
        - CRS: {file_info.get("crs", "Not specified")}
        """
        )
