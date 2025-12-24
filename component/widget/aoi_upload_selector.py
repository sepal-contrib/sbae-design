import logging

import solara
from sepal_ui.solara.components.aoi.aoi_view import AoiView

from component.model import app_state
from component.tile.upload import CurrentFileDisplay, UploadTile
from component.widget.map import SbaeMap

logger = logging.getLogger("sbae.aoi_upload_selector")


@solara.component
def AoiUploadSelector(sbae_map: SbaeMap = None):
    """Component that renders AOI selector or Upload button based on sampling method."""
    show_upload_modal = solara.use_reactive(False)
    sampling_method = app_state.sampling_method.value
    has_uploaded_file = (
        app_state.uploaded_file_info.value is not None
        and app_state.file_path.value is not None
    )

    async def compute_aoi_gdf_worker():
        """Worker function to compute GeoDataFrame from AoiResult in background."""
        if not app_state.aoi_data.value:
            return None

        logger.debug("Starting AOI GeoDataFrame computation")
        app_state.aoi_computing.value = True
        try:
            gdf = await app_state.aoi_data.value.get_gdf_async()
            logger.debug(
                f"Computed GeoDataFrame with {len(gdf) if gdf is not None else 0} features"
            )
            return gdf
        except Exception as e:
            logger.error(f"Error computing AOI GeoDataFrame: {e}")
            return None
        finally:
            app_state.aoi_computing.value = False

    aoi_result = solara.lab.use_task(
        compute_aoi_gdf_worker,
        dependencies=[app_state.aoi_data.value],
    )

    def process_aoi_result():
        """Process AOI computation result."""
        if aoi_result.value is not None:
            app_state.aoi_gdf.value = aoi_result.value
        else:
            app_state.aoi_gdf.value = None

    solara.use_effect(process_aoi_result, [aoi_result.value])

    def cleanup_on_unmount():
        """Cleanup when component unmounts."""

        def cleanup():
            app_state.aoi_computing.value = False

        return cleanup

    solara.use_effect(cleanup_on_unmount, [])

    def open_upload_modal():
        """Open the upload modal."""
        show_upload_modal.value = True

    def close_upload_modal():
        """Close the upload modal."""
        show_upload_modal.value = False

    with solara.Column():
        if sampling_method in ("simple", "systematic"):

            solara.HTML(
                tag="div",
                unsafe_innerHTML="<strong>Select Area of Interest</strong>",
                style="font-size: 16px; margin-bottom: 8px;",
            )

            AoiView(
                value=app_state.aoi_data,
                methods="ALL",
                gee=False,
                map_=sbae_map,
            )

            if app_state.aoi_computing.value:
                solara.Info("⏳ Computing AOI boundaries...")

        elif sampling_method == "stratified":

            if has_uploaded_file:
                CurrentFileDisplay(sbae_map)
            else:
                solara.Text(
                    "For stratified sampling, you need to upload a classification map."
                )

                with solara.Row(justify="center", style={"margin-top": "16px"}):
                    solara.Button(
                        label="Upload Map",
                        icon_name="mdi-upload",
                        on_click=open_upload_modal,
                        color="primary",
                        block=True,
                        small=True,
                    )

    # Render modal outside the Column to prevent unmounting during state changes
    if show_upload_modal.value and sampling_method == "stratified":
        with solara.v.Dialog(
            v_model=show_upload_modal.value,
            on_v_model=show_upload_modal.set,
            max_width="900px",
            persistent=False,
        ):
            with solara.Card(margin=0):

                with solara.Column():
                    UploadTile(sbae_map)

                with solara.CardActions():
                    solara.Button(
                        label="Close",
                        on_click=close_upload_modal,
                        text=True,
                    )
