import base64
import json
import logging
import os
from io import BytesIO

import ipyleaflet
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.warp

# Get a logger for this module
logger = logging.getLogger("sbae")

logger.debug("rasterio and rasterio.warp imported successfully.")

# --- Constants ---
target_crs_epsg = 4326
target_crs_str = f"EPSG:{target_crs_epsg}"


# --- Utility Functions ---
def generate_class_color_map(unique_classes):
    """Generates a unique hex color for each class using a colormap."""
    unique_classes_cleaned = sorted(
        [c for c in unique_classes if pd.notna(c)]
    )  # Renamed to avoid conflict
    n_classes = len(unique_classes_cleaned)
    if n_classes == 0:
        return {}

    if n_classes <= 10:
        cmap_name = "tab10"
    elif n_classes <= 12:
        cmap_name = "Set3"
    elif n_classes <= 20:
        cmap_name = "tab20"
    else:
        cmap_name = "viridis"  # Default for many classes

    try:
        cmap = plt.colormaps[cmap_name]
        colors = [
            matplotlib.colors.to_hex(cmap(i / max(1, n_classes - 1)))
            for i in range(n_classes)
        ]
    except Exception as e:
        logger.warning(f"Colormap '{cmap_name}' failed ({e}). Falling back to viridis.")
        try:
            cmap = plt.colormaps["viridis"]
            colors = [
                matplotlib.colors.to_hex(cmap(i / max(1, n_classes - 1)))
                for i in range(n_classes)
            ]
        except Exception as fallback_e:
            logger.error(
                f"Fallback colormap 'viridis' also failed: {fallback_e}. Returning gray for all classes."
            )
            return {cls: "#808080" for cls in unique_classes_cleaned}

    return {cls: colors[i] for i, cls in enumerate(unique_classes_cleaned)}


# --- Helper Function for Map Overlay (_add_overlay_layer) ---
def _add_overlay_layer(
    map_widget: ipyleaflet.Map,
    map_file_path: str,
    legend_data: pd.DataFrame,  # This is expected to be the raw_area_df or similar
    overlay_bounds,  # For raster: [[lat_min, lon_min], [lat_max, lon_max]]
    vector_gdf,  # For vector: GeoDataFrame
):
    """Adds Raster (ImageOverlay/DataURL) or Vector (GeoJSON) layer.

    Returns the created layer object or None if unsuccessful.
    """
    logger.debug("Attempting to add overlay layer...")

    if not isinstance(map_widget, ipyleaflet.Map):
        logger.error("Invalid map widget provided.")
        return None
    if not map_file_path or not os.path.exists(map_file_path):
        logger.error(f"Invalid map file path provided: {map_file_path}")
        return None

    created_layer = None
    try:
        map_file_str = str(map_file_path)
        basename = os.path.basename(map_file_str)
        raster_extensions = (
            ".tif",
            ".tiff",
            ".img",
            ".pix",
            ".rst",
            ".grd",
            ".vrt",
            ".hdf",
            ".h5",
            ".jpeg2000",
        )
        vector_extensions = (
            ".shp",
            ".sqlite",
            ".gdb",
            ".geojson",
            ".json",
            ".gml",
            ".kml",
            ".tab",
            ".mif",
        )

        if map_file_str.lower().endswith(raster_extensions):
            logger.debug(f"Processing Raster {basename} (ImageOverlay/DataURL)...")
            if (
                overlay_bounds
                and len(overlay_bounds) == 2
                and len(overlay_bounds[0]) == 2
                and len(overlay_bounds[1]) == 2
            ):
                logger.debug(f"Processing Raster {basename} (ImageOverlay/DataURL)...")
                src_raster = None
                data_raster = None
                rgba_image_arr = (
                    None  # Renamed from rgba_image to avoid conflict with plt.imsave
                )
                image_data_url = None  # Renamed from image_url

                try:
                    with rasterio.open(map_file_str) as src_raster:
                        max_dim = 1000  # Max dimension for downsampled display raster
                        h, w = src_raster.height, src_raster.width
                        if h > max_dim or w > max_dim:
                            scale = max_dim / max(h, w)
                            out_h, out_w = int(h * scale), int(w * scale)
                            logger.debug(
                                f"Downsampling {h}x{w} raster to {out_h}x{out_w} for display."
                            )
                            data_raster = src_raster.read(
                                1,  # Assuming single band for classification map
                                out_shape=(out_h, out_w),
                                resampling=rasterio.enums.Resampling.nearest,
                            )
                        else:
                            data_raster = src_raster.read(1)
                            logger.debug("Using full resolution for display.")
                        nodata_val = src_raster.nodata

                    # Colorize raster based on legend_data (e.g., self.model.raw_area_df)
                    if legend_data is not None and "map_code" in legend_data.columns:
                        # Ensure map_code in legend_data is of a comparable type to raster values
                        try:
                            # Attempt to convert map_code to the raster's dtype if necessary
                            # This is a heuristic; raster data types can vary.
                            # For simplicity, assume legend map_code can be compared directly or after basic conversion.
                            # If data_raster.dtype is float, legend_data["map_code"] might need conversion.
                            # If data_raster.dtype is int, ensure legend_data["map_code"] is also int-comparable.
                            legend_codes = (
                                legend_data["map_code"]
                                .dropna()
                                .astype(data_raster.dtype)
                                .unique()
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not reliably convert legend 'map_code' to raster dtype {data_raster.dtype}: {e}. Using original legend codes."
                            )
                            legend_codes = legend_data["map_code"].dropna().unique()

                        raster_codes_present = np.unique(
                            data_raster[~np.isnan(data_raster)]
                        )  # Filter out NaNs if any
                        common_codes = np.intersect1d(
                            legend_codes, raster_codes_present, assume_unique=True
                        )

                        if len(common_codes) > 0:
                            logger.debug(
                                f"Colorizing raster based on {len(common_codes)} common codes found in legend."
                            )
                            color_map_hex = generate_class_color_map(common_codes)

                            def hex_to_rgba_tuple(
                                hex_color_str, alpha=0.7
                            ):  # Renamed var
                                try:
                                    rgb_tuple = matplotlib.colors.to_rgb(
                                        hex_color_str
                                    )  # Renamed var
                                    return tuple(int(c * 255) for c in rgb_tuple) + (
                                        int(alpha * 255),
                                    )
                                except ValueError:
                                    logger.warning(
                                        f"Invalid hex color '{hex_color_str}'. Using transparent gray."
                                    )
                                    return (
                                        128,
                                        128,
                                        128,
                                        0,
                                    )  # Default transparent gray for errors

                            color_map_rgba_tuples = {  # Renamed var
                                code: hex_to_rgba_tuple(hex_c)
                                for code, hex_c in color_map_hex.items()
                            }
                            default_rgba_val = (
                                0,
                                0,
                                0,
                                0,
                            )  # Transparent for unmapped/nodata pixels

                            rgba_image_arr = np.zeros(
                                (data_raster.shape[0], data_raster.shape[1], 4),
                                dtype=np.uint8,
                            )

                            for (
                                code_val,
                                rgba_val,
                            ) in color_map_rgba_tuples.items():  # Renamed vars
                                mask = (data_raster == code_val) & (
                                    ~np.isnan(data_raster)
                                )
                                rgba_image_arr[mask] = rgba_val

                            # Handle nodata explicitly after coloring valid data
                            if nodata_val is not None:
                                if np.isnan(nodata_val):  # If nodata is NaN
                                    nodata_mask = np.isnan(data_raster)
                                else:  # If nodata is a specific value
                                    nodata_mask = data_raster == nodata_val
                                rgba_image_arr[
                                    nodata_mask
                                ] = default_rgba_val  # Make nodata transparent

                            logger.debug("Converting colored array to PNG bytes...")
                            buffer = BytesIO()
                            plt.imsave(
                                buffer, rgba_image_arr, format="png"
                            )  # plt from import
                            buffer.seek(0)
                            png_bytes = buffer.read()
                            buffer.close()
                            logger.debug(f"Generated PNG ({len(png_bytes)} bytes).")

                            png_base64 = base64.b64encode(png_bytes).decode("utf-8")
                            image_data_url = f"data:image/png;base64,{png_base64}"
                            logger.debug(
                                f"Created Data URL (approx len: {len(image_data_url)})."
                            )
                        else:
                            logger.warning(
                                "No common codes found between raster data and legend. Cannot colorize raster for display."
                            )
                    else:
                        logger.warning(
                            "Legend data or 'map_code' column missing. Cannot colorize raster for display."
                        )

                    if (
                        image_data_url
                    ):  # Check if image_data_url was successfully created
                        logger.debug("Adding ImageOverlay to map...")
                        img_overlay = ipyleaflet.ImageOverlay(
                            url=image_data_url,
                            bounds=overlay_bounds,  # [[south, west], [north, east]]
                            name=f"Raster: {basename}",
                        )
                        map_widget.add_layer(img_overlay)
                        logger.debug("Successfully ADDED ImageOverlay.")
                        created_layer = img_overlay
                    else:
                        logger.warning(
                            "No image URL generated (e.g. due to missing colors or common codes), skipping ImageOverlay."
                        )

                except MemoryError:
                    logger.error(
                        "Insufficient memory to process raster for display. Try a smaller file or increase available memory."
                    )
                except Exception:
                    logger.exception(f"Error processing Raster Overlay for {basename}:")
                finally:
                    if "data_raster" in locals() and data_raster is not None:
                        del data_raster
                    if "rgba_image_arr" in locals() and rgba_image_arr is not None:
                        del rgba_image_arr
            else:
                logger.warning(
                    "Skipping Raster overlay - valid EPSG:4326 bounds were not provided or incorrect format."
                )

        elif map_file_str.lower().endswith(vector_extensions):
            if vector_gdf is not None and not vector_gdf.empty:
                logger.debug(f"Processing Vector Overlay for {basename}...")
                try:
                    # Ensure vector_gdf is in EPSG:4326 if it's going on a standard web map
                    # This should ideally be handled by the caller ensuring vector_gdf is map-ready.
                    # For this function, we assume it's already in the correct CRS for ipyleaflet.

                    geom_col_name = vector_gdf.geometry.name  # Renamed
                    attr_cols = [c for c in vector_gdf.columns if c != geom_col_name]
                    default_style_dict = {  # Renamed
                        "color": "black",
                        "weight": 1,
                        "fillColor": "#808080",
                        "fillOpacity": 0.6,
                    }
                    default_hover_style = {
                        "fillColor": "red",
                        "fillOpacity": 0.8,
                    }  # Renamed
                    style_callback = None  # Renamed
                    layer_display_name = f"Vector: {basename}"  # Renamed

                    # Attempt to style based on the first attribute column and legend_data
                    if attr_cols:
                        map_code_col_for_style = attr_cols[
                            0
                        ]  # Use local var for clarity
                        logger.debug(
                            f"Using vector column '{map_code_col_for_style}' for styling attempts."
                        )

                        if map_code_col_for_style in vector_gdf.columns:
                            if (
                                legend_data is not None
                                and "map_code" in legend_data.columns
                            ):
                                # Attempt to match types between vector attribute and legend's map_code
                                try:
                                    vec_attr_type = vector_gdf[
                                        map_code_col_for_style
                                    ].dtype
                                    legend_codes_for_vec = (
                                        legend_data["map_code"]
                                        .dropna()
                                        .astype(vec_attr_type)
                                        .unique()
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not reliably convert legend 'map_code' to vector attribute type {vec_attr_type} for styling: {e}. Using original legend codes."
                                    )
                                    legend_codes_for_vec = (
                                        legend_data["map_code"].dropna().unique()
                                    )

                                vector_codes_in_gdf = (
                                    vector_gdf[map_code_col_for_style].dropna().unique()
                                )
                                common_codes_for_vec_style = np.intersect1d(
                                    legend_codes_for_vec,
                                    vector_codes_in_gdf,
                                    assume_unique=True,
                                )

                                if len(common_codes_for_vec_style) > 0:
                                    logger.debug(
                                        f"Colorizing vector based on {len(common_codes_for_vec_style)} common codes from attribute '{map_code_col_for_style}'."
                                    )
                                    color_map_hex_vec = generate_class_color_map(
                                        common_codes_for_vec_style
                                    )
                                    color_lookup = {  # Renamed
                                        code: hex_c
                                        for code, hex_c in color_map_hex_vec.items()
                                    }
                                    fallback_fill_color = "#808080"  # Renamed

                                    def dynamic_style_callback(
                                        feature,
                                        cmap=color_lookup,
                                        code_col=map_code_col_for_style,
                                        fallback=fallback_fill_color,
                                    ):
                                        code_value = feature["properties"].get(code_col)
                                        fill = cmap.get(code_value, fallback)
                                        return {
                                            "fillColor": fill,
                                            "color": "black",
                                            "weight": 1,
                                            "fillOpacity": 0.7,
                                        }

                                    style_callback = dynamic_style_callback
                                    layer_display_name = f"Vector: {basename} (Styled by {map_code_col_for_style})"
                                else:
                                    logger.warning(
                                        f"No common codes found for vector styling using attribute '{map_code_col_for_style}'. Using default style."
                                    )
                            else:
                                logger.warning(
                                    "Legend data or 'map_code' missing for vector coloring. Using default style."
                                )
                        else:
                            logger.warning(
                                f"Styling column '{map_code_col_for_style}' not found in vector attributes. Using default style."
                            )
                    else:
                        logger.warning(
                            "Vector has no attribute columns for styling. Using default style."
                        )

                    logger.debug("Converting GDF to GeoJSON dictionary...")
                    # Clean geometries before converting to JSON
                    vector_gdf_cleaned = vector_gdf[
                        vector_gdf.is_valid & ~vector_gdf.is_empty
                    ].copy()  # Use .copy() to avoid SettingWithCopyWarning
                    if vector_gdf_cleaned.empty:
                        logger.warning(
                            "No valid geometries remain after cleaning GDF. Cannot create GeoJSON layer."
                        )
                        return None

                    geojson_data_dict = json.loads(
                        vector_gdf_cleaned.to_json()
                    )  # Renamed
                    logger.debug("Creating GeoJSON layer...")

                    vector_layer_obj = ipyleaflet.GeoJSON(  # Renamed
                        data=geojson_data_dict,
                        style_callback=style_callback,  # Use the dynamically created callback or None
                        style=default_style_dict,  # Default style if callback is None or not styling features
                        hover_style=default_hover_style,
                        name=layer_display_name,
                    )
                    map_widget.add_layer(vector_layer_obj)
                    logger.debug(
                        f"Successfully ADDED GeoJSON layer '{layer_display_name}'."
                    )
                    created_layer = vector_layer_obj

                except Exception:
                    logger.exception(
                        f"Error creating/adding GeoJSON layer for {basename}:"
                    )
            else:
                logger.warning(
                    "Skipping Vector overlay - valid GeoDataFrame was not provided or was empty."
                )
        else:
            logger.warning(f"File type not recognized for overlay: {basename}")

    except Exception:
        logger.exception(
            f"An unexpected error occurred during overlay processing for {map_file_path}:"
        )

    if created_layer:
        logger.debug(
            f"Overlay layer object '{getattr(created_layer, 'name', 'Unnamed Layer')}' CREATED."
        )
    else:
        logger.debug("Overlay layer object was NOT created.")
    return created_layer
