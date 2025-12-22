import logging

import geopandas as gpd
import solara
from sepal_ui.solara.components.aoi import AoiView

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

    async def compute_gee_features_worker():
        """Worker function to compute GEE FeatureCollection in background."""
        feature_collection = (
            app_state.aoi_data.value.feature_collection
            if app_state.aoi_data.value
            and app_state.aoi_data.value.feature_collection is not None
            else None
        )

        if not feature_collection:
            return None

        if not sbae_map or not hasattr(sbae_map, "gee_interface"):
            logger.warning("No gee_interface available")
            return None

        logger.debug("Starting GEE FeatureCollection computation")
        app_state.aoi_computing.value = True
        try:
            info = await sbae_map.gee_interface.get_info_async(feature_collection)
            features = info.get("features", [])
            logger.debug(f"Computed {len(features)} features from GEE")
            return features
        except Exception as e:
            logger.error(f"Error computing GEE features: {e}")
            return None
        finally:
            app_state.aoi_computing.value = False

    gee_result = solara.lab.use_task(
        compute_gee_features_worker,
        dependencies=[app_state.aoi_data.value],
    )

    def process_gee_result():
        """Process GEE computation result and create GeoDataFrame."""
        if gee_result.value is not None and gee_result.value:
            try:
                gdf = gpd.GeoDataFrame.from_features(gee_result.value, crs="EPSG:4326")
                app_state.aoi_gdf.value = gdf
                logger.debug(f"Created GeoDataFrame with {len(gdf)} features")
            except Exception as e:
                logger.error(f"Error creating GeoDataFrame: {e}")
                app_state.aoi_gdf.value = None

    solara.use_effect(process_gee_result, [gee_result.value])

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
                gee=True,
                map_=sbae_map,
            )

            if app_state.aoi_computing.value:
                solara.Info("⏳ Computing AOI boundaries...")

        elif sampling_method == "stratified":

            if has_uploaded_file:
                CurrentFileDisplay(sbae_map)
            else:
                solara.Markdown(
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
