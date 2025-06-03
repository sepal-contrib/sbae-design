# component/scripts/processing.py

import logging
import os
import random
from math import ceil
from pathlib import Path

import geopandas as gpd
import ipywidgets as widgets
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from pyproj import CRS, Transformer
from scipy.special import ndtri
from shapely.geometry import Point

logger = logging.getLogger("sbae")
logger.debug("pyproj imported successfully.")

# --- Constants ---
target_crs_epsg = 4326
target_crs_str = f"EPSG:{target_crs_epsg}"


# --- Globals used by run_sample_design ---
sample_size_output: widgets.Output = None
sample_points_df: pd.DataFrame = None
objective_widget = None
target_class_widget = None
target_class_allowable_error_widget = None
expected_user_accuracy_widgets = None
allocation_method_widget = None
target_overall_accuracy_widget = None
allowable_error_widget = None
confidence_level_widget = None
min_samples_per_class_widget = None
# --- End Globals ---


def get_z_score(confidence_level):
    if confidence_level == 0.90:
        return 1.645
    if confidence_level == 0.95:
        return 1.960
    if confidence_level == 0.99:
        return 2.576

    try:
        p_value = (1 + confidence_level) / 2.0
        if not (0 < p_value < 1):
            raise ValueError(
                f"Calculated probability {p_value:.4f} for ndtri is out of (0,1) range from confidence level {confidence_level:.3f}."
            )
        return ndtri(p_value)
    except Exception as e:
        logger.error(
            f"Error calculating Z-score with scipy for confidence {confidence_level:.3f}: {e}"
        )
        raise ValueError(
            f"Could not calculate Z-score for {confidence_level:.3f} using scipy. Check input or library."
        )


def get_output_dir(input_file_path_str: str):
    logger.debug(f"get_output_dir called for {input_file_path_str}")
    if (
        input_file_path_str
        and isinstance(input_file_path_str, (str, Path))
        and Path(input_file_path_str).is_file()
    ):
        try:
            p = Path(input_file_path_str)
            dirname = p.parent
            basename_no_ext = p.stem
            safe_basename = "".join(c if c.isalnum() else "_" for c in basename_no_ext)
            if not safe_basename:
                safe_basename = "file"

            subdir_name = f"sae_design_{safe_basename}"
            output_dir_path = dirname / subdir_name
            output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory set to: {output_dir_path}")
            return str(output_dir_path)
        except Exception:
            logger.exception(
                f"Error creating output directory for {input_file_path_str}:"
            )
            return None
    else:
        logger.error(
            f"Invalid input file path for get_output_dir: {input_file_path_str}"
        )
        return None


def compute_map_area(input_map_file_path_str: str):
    logger.info(f"Computing map area for: {input_map_file_path_str}")
    area_df = None
    if (
        not input_map_file_path_str
        or not isinstance(input_map_file_path_str, (str, Path))
        or not Path(input_map_file_path_str).exists()
    ):
        logger.error(
            f"Invalid file path for area computation: {input_map_file_path_str}"
        )
        return None

    file_path_obj = Path(input_map_file_path_str)
    basename = file_path_obj.name
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

    if file_path_obj.suffix.lower() in raster_extensions:
        logger.info(f"Processing '{basename}' as Raster for area computation...")
        try:
            with rasterio.open(file_path_obj) as src:
                crs = src.crs
                logger.info(f"Raster CRS: {crs}")
                if crs and crs.is_geographic:
                    logger.warning(
                        "Raster CRS is geographic. Area units will be in square degrees which may not be directly comparable for stratified sampling unless pixel counts are used."
                    )
                elif not crs:
                    logger.warning(
                        "No CRS defined for raster. Assuming pixel areas are in consistent, albeit unknown, projected units."
                    )

                if (
                    not src.res
                    or len(src.res) < 2
                    or not all(src.res)
                    or src.res[0] == 0
                    or src.res[1] == 0
                ):
                    logger.error(
                        f"Invalid or zero resolution found in raster: {src.res}. Cannot compute area."
                    )
                    return None

                pixel_area_map_units = abs(src.res[0] * src.res[1])
                logger.debug(f"Pixel area in map units: {pixel_area_map_units}")

                data = src.read(1)
                nodata_val = src.nodata
                logger.debug(f"Nodata value from raster: {nodata_val}")

                unique_values, counts = np.unique(data, return_counts=True)

                valid_mask = np.ones_like(unique_values, dtype=bool)
                if nodata_val is not None:
                    if np.isnan(nodata_val):
                        valid_mask &= ~np.isnan(unique_values)
                    else:
                        valid_mask &= unique_values != nodata_val
                valid_mask &= ~np.isnan(unique_values)

                unique_values_filtered = unique_values[valid_mask]
                counts_filtered = counts[valid_mask]

                if unique_values_filtered.size == 0:
                    logger.warning(
                        "No valid class values found in raster after filtering nodata/NaN."
                    )
                    return None

                area_df = pd.DataFrame(
                    {
                        "map_code": unique_values_filtered,
                        "map_area": counts_filtered * pixel_area_map_units,
                    }
                )
                area_df = area_df[area_df["map_area"] > 1e-9].reset_index(drop=True)

                if area_df.empty:
                    logger.warning("No classes with positive area found in raster.")
                    return None
                logger.info(f"Computed areas for {len(area_df)} raster classes.")
        except Exception:
            logger.exception(
                f"Error reading/processing raster '{basename}' for area computation:"
            )
            return None

    elif file_path_obj.suffix.lower() in vector_extensions:
        logger.info(f"Processing '{basename}' as Vector for area computation...")
        try:
            gdf = gpd.read_file(file_path_obj)
            if gdf.empty:
                logger.warning("Vector file is empty. No areas to compute.")
                return None

            geom_col_name = gdf.geometry.name
            if not (
                geom_col_name in gdf.columns
                and not gdf[geom_col_name].isnull().all()
                and not gdf[geom_col_name].is_empty.all()
            ):
                logger.error("No valid geometries found in vector file.")
                return None

            crs = gdf.crs
            logger.info(f"Vector CRS: {crs}")
            if crs and crs.is_geographic:
                logger.warning(
                    "Vector CRS is geographic. Area units will be in square degrees. Consider reprojecting to an equal-area projection for meaningful area calculation if needed for sampling weights."
                )
            elif not crs:
                logger.warning(
                    "No CRS defined for vector. Area calculation will assume planar geometry with unknown units."
                )

            attribute_cols = [col for col in gdf.columns if col != geom_col_name]
            if not attribute_cols:
                logger.error(
                    "No attribute columns found in vector file to use as class identifier."
                )
                return None

            map_code_col_name = attribute_cols[0]
            logger.info(
                f"Using attribute column '{map_code_col_name}' from vector as map code for area aggregation."
            )
            if not (
                map_code_col_name in gdf.columns
                and not gdf[map_code_col_name].isnull().all()
            ):
                logger.error(
                    f"Map code column '{map_code_col_name}' is invalid or all null."
                )
                return None

            try:
                gdf[geom_col_name] = gdf[geom_col_name].buffer(0)
                gdf_valid = gdf[
                    gdf[geom_col_name].is_valid & ~gdf[geom_col_name].is_empty
                ].copy()
                if gdf_valid.empty:
                    logger.warning(
                        "No valid geometries remain after cleaning. No areas to compute."
                    )
                    return None

                gdf_valid["map_area_calc"] = gdf_valid.geometry.area
                gdf_valid["map_area_calc"] = (
                    gdf_valid["map_area_calc"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )

                gdf_filtered = gdf_valid[
                    (gdf_valid["map_area_calc"] > 1e-9)
                    & (gdf_valid[map_code_col_name].notna())
                ].reset_index(drop=True)
                if gdf_filtered.empty:
                    logger.warning(
                        "No valid features with positive area and map code found after filtering."
                    )
                    return None

            except Exception:
                logger.exception("Error during geometry area calculation or cleaning:")
                return None

            area_df = (
                gdf_filtered.groupby(map_code_col_name)["map_area_calc"]
                .sum()
                .reset_index()
            )
            area_df.columns = ["map_code", "map_area"]
            area_df = area_df[area_df["map_area"] > 1e-9].reset_index(drop=True)

            if area_df.empty:
                logger.warning(
                    "No classes with positive total area found in vector after grouping."
                )
                return None
            logger.info(f"Computed areas for {len(area_df)} vector classes.")

        except Exception:
            logger.exception(
                f"Error reading/processing vector '{basename}' for area computation:"
            )
            return None
    else:
        logger.error(f"Unsupported file format for area computation: {basename}")
        return None

    if area_df is not None and "map_edited_class" not in area_df.columns:
        area_df["map_edited_class"] = area_df["map_code"].astype(str)

    return area_df


def generate_sample_points(
    map_file_str, final_area_df_class_lookup, n_samples_per_class_dict
):
    logger.info("--- Running generate_sample_points ---")
    sample_points_list = []

    if final_area_df_class_lookup is None or not all(
        c in final_area_df_class_lookup.columns
        for c in ["map_code", "map_edited_class"]
    ):
        logger.error(
            "Final area DataFrame for sample generation is invalid or missing required columns ('map_code', 'map_edited_class')."
        )
        return pd.DataFrame()

    class_lookup_dict = final_area_df_class_lookup.set_index("map_code")[
        "map_edited_class"
    ].to_dict()

    map_file_path_obj = Path(map_file_str)
    basename = map_file_path_obj.name
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

    try:
        if map_file_path_obj.suffix.lower() in raster_extensions:
            logger.info(f"Generating samples from Raster: {basename}")

            with rasterio.open(map_file_path_obj) as src:
                transform_affine = src.transform
                data = src.read(1)
                source_crs_obj = src.crs
                logger.info(f"Raster source CRS: {source_crs_obj}")

                transformer_to_wgs84 = None
                if source_crs_obj:
                    source_pyproj_crs = CRS.from_user_input(source_crs_obj)
                    if source_pyproj_crs.to_epsg() != target_crs_epsg:
                        try:
                            transformer_to_wgs84 = Transformer.from_crs(
                                source_pyproj_crs,
                                f"EPSG:{target_crs_epsg}",
                                always_xy=True,
                            )
                            logger.debug(
                                f"Reprojection transformer created for raster: {source_pyproj_crs.to_string()} -> EPSG:{target_crs_epsg}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to create CRS transformer for raster: {e}. Points will be in original CRS if any."
                            )
                    else:
                        logger.debug(
                            "Raster is already in target CRS EPSG:4326. No transformation needed."
                        )
                elif not source_crs_obj:
                    logger.warning(
                        "No CRS defined for raster. Assuming output coordinates are EPSG:4326 if no transformer is active, which might be incorrect."
                    )

                for map_code_val, n_samples_req in n_samples_per_class_dict.items():
                    if n_samples_req <= 0:
                        continue

                    edited_class_name = class_lookup_dict.get(
                        map_code_val, f"Unknown Code {map_code_val}"
                    )

                    try:
                        map_code_val_typed = np.array([map_code_val]).astype(
                            data.dtype
                        )[0]
                    except Exception as e:
                        logger.warning(
                            f"Could not convert map_code {map_code_val} to raster dtype {data.dtype} for class '{edited_class_name}'. Skipping. Error: {e}"
                        )
                        continue

                    rows, cols = np.where(data == map_code_val_typed)
                    num_available_pixels = len(rows)

                    if num_available_pixels == 0:
                        logger.warning(
                            f"No pixels found for map code {map_code_val_typed} ('{edited_class_name}'). Cannot generate samples for this class."
                        )
                        continue

                    num_samples_to_gen = min(n_samples_req, num_available_pixels)
                    if num_samples_to_gen < n_samples_req:
                        logger.warning(
                            f"Requested {n_samples_req} samples for class '{edited_class_name}' (Code: {map_code_val_typed}), "
                            f"but only {num_available_pixels} pixels are available. Generating {num_samples_to_gen}."
                        )

                    sampled_indices_arr = random.sample(
                        range(num_available_pixels), num_samples_to_gen
                    )
                    sampled_rows, sampled_cols = (
                        rows[sampled_indices_arr],
                        cols[sampled_indices_arr],
                    )

                    for r_idx, c_idx in zip(sampled_rows, sampled_cols):
                        x_coord_orig, y_coord_orig = transform_affine * (
                            c_idx + 0.5,
                            r_idx + 0.5,
                        )

                        lon_out, lat_out = x_coord_orig, y_coord_orig
                        if transformer_to_wgs84:
                            try:
                                lon_out, lat_out = transformer_to_wgs84.transform(
                                    x_coord_orig, y_coord_orig
                                )
                            except Exception as proj_error:
                                logger.warning(
                                    f"Reprojection failed for raster point (orig: {x_coord_orig:.2f},{y_coord_orig:.2f}) "
                                    f"for class '{edited_class_name}'. Skipping point. Error: {proj_error}"
                                )
                                continue

                        sample_points_list.append(
                            {
                                "latitude": lat_out,
                                "longitude": lon_out,
                                "map_code": map_code_val,
                                "map_edited_class": edited_class_name,
                            }
                        )
                    logger.info(
                        f"Generated {num_samples_to_gen} points for class '{edited_class_name}' (Raster Code: {map_code_val_typed})"
                    )
                del data

        elif map_file_path_obj.suffix.lower() in vector_extensions:
            logger.info(f"Generating samples from Vector: {basename}")

            gdf = gpd.read_file(map_file_path_obj)
            if gdf.empty:
                logger.error("Input vector file for sample generation is empty.")
                return pd.DataFrame()

            source_crs_obj_vec = gdf.crs
            logger.info(f"Vector source CRS: {source_crs_obj_vec}")
            geom_col_name_vec = gdf.geometry.name

            attribute_cols_vec = [
                col for col in gdf.columns if col != geom_col_name_vec
            ]
            if not attribute_cols_vec:
                logger.error(
                    "Vector file has no attribute columns to identify classes for sampling."
                )
                return pd.DataFrame()

            map_code_col_vec = attribute_cols_vec[0]
            logger.info(
                f"Using attribute '{map_code_col_vec}' from vector as class identifier for sampling."
            )
            if map_code_col_vec not in gdf.columns:
                logger.error(
                    f"Map code column '{map_code_col_vec}' not found in vector attributes."
                )
                return pd.DataFrame()

            transformer_vec_to_wgs84 = None
            if source_crs_obj_vec:
                source_pyproj_crs_vec = CRS.from_user_input(source_crs_obj_vec)
                if source_pyproj_crs_vec.to_epsg() != target_crs_epsg:
                    try:
                        transformer_vec_to_wgs84 = Transformer.from_crs(
                            source_pyproj_crs_vec,
                            f"EPSG:{target_crs_epsg}",
                            always_xy=True,
                        )
                        logger.debug(
                            f"Reprojection transformer created for vector: {source_pyproj_crs_vec.to_string()} -> EPSG:{target_crs_epsg}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create CRS transformer for vector: {e}. Points will be in original CRS if any."
                        )
                else:
                    logger.debug(
                        "Vector is already in target CRS EPSG:4326. No transformation needed."
                    )
            elif not source_crs_obj_vec:
                logger.warning(
                    "No CRS defined for vector. Assuming output coordinates are EPSG:4326 if no transformer, which might be incorrect."
                )

            for (
                map_code_val_dict,
                n_samples_req_dict,
            ) in n_samples_per_class_dict.items():
                if n_samples_req_dict <= 0:
                    continue

                edited_class_name_dict = class_lookup_dict.get(
                    map_code_val_dict, f"Unknown Code {map_code_val_dict}"
                )

                try:
                    gdf_class_code_type = gdf[map_code_col_vec].dtype
                    map_code_val_dict_typed = np.array([map_code_val_dict]).astype(
                        gdf_class_code_type
                    )[0]
                except Exception as e:
                    logger.warning(
                        f"Could not convert map_code {map_code_val_dict} to vector attribute type {gdf_class_code_type} for class '{edited_class_name_dict}'. Skipping. Error: {e}"
                    )
                    continue

                class_gdf_filtered = gdf[
                    gdf[map_code_col_vec] == map_code_val_dict_typed
                ].copy()

                class_gdf_filtered[geom_col_name_vec] = class_gdf_filtered[
                    geom_col_name_vec
                ].buffer(0)
                class_gdf_filtered = class_gdf_filtered[
                    class_gdf_filtered[geom_col_name_vec].is_valid
                    & ~class_gdf_filtered[geom_col_name_vec].is_empty
                ]

                if class_gdf_filtered.empty:
                    logger.warning(
                        f"No valid/non-empty geometries for map code {map_code_val_dict_typed} ('{edited_class_name_dict}') in vector. Cannot generate samples."
                    )
                    continue

                minx, miny, maxx, maxy = class_gdf_filtered.total_bounds
                if not (
                    np.all(np.isfinite([minx, miny, maxx, maxy]))
                    and minx <= maxx
                    and miny <= maxy
                ):
                    logger.warning(
                        f"Invalid or illogical bounds {minx, miny, maxx, maxy} for class '{edited_class_name_dict}' (Vector). Skipping sampling for this class."
                    )
                    continue

                samples_generated_count = 0
                max_attempts_per_point = n_samples_req_dict * 100

                logger.info(
                    f"Attempting to generate {n_samples_req_dict} points for class '{edited_class_name_dict}' (Vector Code: {map_code_val_dict_typed})..."
                )

                for _ in range(max_attempts_per_point):
                    if samples_generated_count >= n_samples_req_dict:
                        break

                    rand_x_orig = random.uniform(minx, maxx)
                    rand_y_orig = random.uniform(miny, maxy)
                    point_geom = Point(rand_x_orig, rand_y_orig)

                    if class_gdf_filtered.contains(point_geom).any():
                        lon_out_vec, lat_out_vec = rand_x_orig, rand_y_orig
                        if transformer_vec_to_wgs84:
                            try:
                                (
                                    lon_out_vec,
                                    lat_out_vec,
                                ) = transformer_vec_to_wgs84.transform(
                                    rand_x_orig, rand_y_orig
                                )
                            except Exception as proj_error:
                                logger.warning(
                                    f"Reprojection failed for vector point (orig: {rand_x_orig:.2f},{rand_y_orig:.2f}) "
                                    f"for class '{edited_class_name_dict}'. Skipping point. Error: {proj_error}"
                                )
                                continue

                        sample_points_list.append(
                            {
                                "latitude": lat_out_vec,
                                "longitude": lon_out_vec,
                                "map_code": map_code_val_dict,
                                "map_edited_class": edited_class_name_dict,
                            }
                        )
                        samples_generated_count += 1

                logger.info(
                    f"Generated {samples_generated_count}/{n_samples_req_dict} points for class '{edited_class_name_dict}' (Vector Code: {map_code_val_dict_typed})."
                )
                if samples_generated_count < n_samples_req_dict:
                    logger.warning(
                        f"Low generation rate for class '{edited_class_name_dict}' (Vector). "
                        f"Generated {samples_generated_count} of {n_samples_req_dict} requested samples after {max_attempts_per_point} total attempts for class."
                    )
            del gdf, class_gdf_filtered
        else:
            logger.error(f"Unsupported file format for sample generation: {basename}")
    except Exception:
        logger.exception("An unexpected error occurred during sample point generation:")

    logger.info(
        f"--- generate_sample_points finished. Generated {len(sample_points_list)} points total. ---"
    )
    return pd.DataFrame(sample_points_list) if sample_points_list else pd.DataFrame()


def output_sample_points(df_samples_to_output: pd.DataFrame, output_dir_path_str: str):
    global sample_size_output
    output_widget_ref = sample_size_output

    def ui_message(msg_text, log_as_info=True):
        if log_as_info:
            logger.info(msg_text)
        else:
            logger.debug(msg_text)

        if output_widget_ref:
            if hasattr(output_widget_ref, "add_msg"):
                output_widget_ref.add_msg(str(msg_text) + "\n")
            elif isinstance(output_widget_ref, widgets.Output):
                with output_widget_ref:
                    print(msg_text)
        else:
            print(msg_text)

    ui_message("--- Outputting Sample Points ---")
    if df_samples_to_output is None or df_samples_to_output.empty:
        ui_message("No sample points to save.", log_as_info=False)
        logger.warning("output_sample_points called with no samples.")
        return
    if not output_dir_path_str or not os.path.isdir(output_dir_path_str):
        err_msg = (
            f"Output directory '{output_dir_path_str}' is invalid for saving samples."
        )
        ui_message(err_msg, log_as_info=False)
        logger.error(err_msg)
        return

    csv_filepath_obj = Path(output_dir_path_str) / "sample_points.csv"
    try:
        df_samples_to_output.to_csv(csv_filepath_obj, index=False)
        ui_message(f"Sample points saved to CSV: {csv_filepath_obj}")
    except Exception as e:
        err_msg = f"Error saving sample points to CSV ({csv_filepath_obj}): {e}"
        ui_message(err_msg, log_as_info=False)
        logger.exception(f"Failed to save CSV output to {csv_filepath_obj}:")

    geojson_filepath_obj = Path(output_dir_path_str) / "sample_points.geojson"
    try:
        if not all(
            c in df_samples_to_output.columns for c in ["latitude", "longitude"]
        ):
            err_msg = "Missing 'latitude' or 'longitude' columns in sample points DataFrame. Cannot create GeoJSON."
            ui_message(err_msg, log_as_info=False)
            logger.error(err_msg)
            return

        df_copy_for_geojson = df_samples_to_output.copy()
        df_copy_for_geojson["latitude"] = pd.to_numeric(
            df_copy_for_geojson["latitude"], errors="coerce"
        )
        df_copy_for_geojson["longitude"] = pd.to_numeric(
            df_copy_for_geojson["longitude"], errors="coerce"
        )

        valid_points_for_geojson = df_copy_for_geojson.dropna(
            subset=["latitude", "longitude"]
        )

        if valid_points_for_geojson.empty:
            warn_msg = "No valid coordinates found for GeoJSON output after cleaning. GeoJSON file will not be created."
            ui_message(warn_msg, log_as_info=False)
            logger.warning(warn_msg)
            return

        gdf_samples_out = gpd.GeoDataFrame(
            valid_points_for_geojson,
            geometry=gpd.points_from_xy(
                valid_points_for_geojson.longitude, valid_points_for_geojson.latitude
            ),
            crs=f"EPSG:{target_crs_epsg}",
        )
        gdf_samples_out.to_file(geojson_filepath_obj, driver="GeoJSON")
        ui_message(f"Sample points saved to GeoJSON: {geojson_filepath_obj}")
    except Exception as e:
        err_msg = f"Error saving sample points to GeoJSON ({geojson_filepath_obj}): {e}"
        ui_message(err_msg, log_as_info=False)
        logger.exception(f"Failed to save GeoJSON output to {geojson_filepath_obj}:")


def run_sample_design(
    map_file_path_str_arg: str, final_area_df_arg: pd.DataFrame, output_dir_str_arg: str
):
    global sample_points_df, sample_size_output
    global objective_widget, target_class_widget, target_class_allowable_error_widget
    global expected_user_accuracy_widgets, allocation_method_widget
    global target_overall_accuracy_widget, allowable_error_widget
    global confidence_level_widget, min_samples_per_class_widget

    sample_points_df = None
    logger.info(">>> Entered run_sample_design function.")

    output_widget_ref_local = sample_size_output

    def report_to_ui_and_log(message_text, ui_level="info", log_level="info"):
        if hasattr(logger, log_level):
            getattr(logger, log_level)(message_text)
        else:
            logger.info(message_text)

        if output_widget_ref_local:
            if (
                hasattr(output_widget_ref_local, "reset")
                and hasattr(output_widget_ref_local, "add_msg")
                and hasattr(output_widget_ref_local, "type")
            ):
                if ui_level == "summary_start":
                    output_widget_ref_local.reset()
                    output_widget_ref_local.type = "info"
                    output_widget_ref_local.add_msg(str(message_text) + "\n")
                elif ui_level == "summary_end":
                    output_widget_ref_local.type = message_text.get("type", "success")
                    output_widget_ref_local.add_msg(
                        str(message_text.get("msg", "")) + "\n"
                    )
                else:
                    output_widget_ref_local.type = (
                        ui_level
                        if ui_level in ["info", "warning", "error", "success"]
                        else "info"
                    )
                    output_widget_ref_local.add_msg(str(message_text) + "\n")
                if hasattr(output_widget_ref_local, "show"):
                    output_widget_ref_local.show()

            elif isinstance(output_widget_ref_local, widgets.Output):
                with output_widget_ref_local:
                    if ui_level == "summary_start":
                        from IPython.display import clear_output

                        clear_output(wait=True)
                    print(
                        f"[{ui_level.upper() if isinstance(message_text, str) else 'INFO'}] {message_text}"
                    )
        else:
            print(f"NO_UI_WIDGET_LOG [{log_level.upper()}]: {message_text}")

    report_to_ui_and_log(
        "--- Running Sample Design Calculation ---", ui_level="summary_start"
    )

    results_dict = {
        "success": False,
        "actual_total_samples": None,
        "summary_df": None,
        "generated_samples_df": None,
        "output_directory": output_dir_str_arg,
        "summary_text": "",
        "error_message": None,
    }

    try:
        report_to_ui_and_log("Retrieving parameters...", log_level="debug")
        objective = objective_widget.v_model
        target_oa_param = target_overall_accuracy_widget.v_model
        allowable_err_overall = allowable_error_widget.v_model
        confidence_param = confidence_level_widget.v_model
        min_n_per_class = min_samples_per_class_widget.v_model
        alloc_method_param = allocation_method_widget.v_model

        z_score_val = get_z_score(confidence_param)

        if not (0 < target_oa_param < 1):
            raise ValueError("Target OA must be between 0 and 1.")
        if not (0 < allowable_err_overall < 1):
            raise ValueError("Allowable Error (Overall) must be between 0 and 1.")
        if not (0.8 <= confidence_param < 1):
            raise ValueError("Confidence Level must be between 0.8 and 1 (e.g. 0.999).")
        if not (min_n_per_class > 0):
            raise ValueError("Min Samples per Class must be positive.")
        logger.debug(
            f"Parameters: Objective={objective}, TargetOA={target_oa_param}, AllowableErrOverall={allowable_err_overall}, Confidence={confidence_param}, MinNClass={min_n_per_class}, AllocMethod={alloc_method_param}, Z-score={z_score_val:.3f}"
        )

        target_class_code_val, n_req_target_class, target_class_name_display = (
            None,
            0,
            "N/A",
        )
        if objective == "Target Class Precision":
            target_class_code_val = target_class_widget.v_model
            if target_class_code_val is None:
                raise ValueError(
                    "Please select the Target Class for 'Target Class Precision' objective."
                )

            target_class_name_display = next(
                (
                    opt["text"]
                    for opt in target_class_widget.items
                    if opt["value"] == target_class_code_val
                ),
                f"Class Code {target_class_code_val}",
            )
            target_class_err_param = target_class_allowable_error_widget.v_model
            if not (0 < target_class_err_param < 1):
                raise ValueError(
                    "Target Class Allowable Error must be between 0 and 1."
                )

            if not (
                expected_user_accuracy_widgets
                and target_class_code_val in expected_user_accuracy_widgets
            ):
                raise ValueError(
                    f"Expected User Accuracy widget missing for target class '{target_class_name_display}' (Code: {target_class_code_val})."
                )

            target_class_ua_param = expected_user_accuracy_widgets[
                target_class_code_val
            ].v_model
            if not (0 < target_class_ua_param < 1):
                raise ValueError(
                    f"Expected UA for '{target_class_name_display}' (Code {target_class_code_val}) must be between 0 and 1."
                )

            numerator_tc = (
                z_score_val**2 * target_class_ua_param * (1 - target_class_ua_param)
            )
            denominator_tc = target_class_err_param**2
            n_req_target_class = (
                ceil(numerator_tc / denominator_tc)
                if numerator_tc > 0 and denominator_tc > 0
                else 0
            )
            n_req_target_class = max(n_req_target_class, min_n_per_class)
            logger.debug(
                f"Target Class '{target_class_name_display}': UA={target_class_ua_param}, Err={target_class_err_param}, RequiredN={n_req_target_class}"
            )

        report_to_ui_and_log("Preparing area data...", log_level="debug")
        if final_area_df_arg is None or final_area_df_arg.empty:
            raise ValueError(
                "Final class area data (final_area_df_arg) is missing or empty."
            )

        design_df_local = final_area_df_arg.copy()
        req_cols = ["map_code", "map_area", "map_edited_class"]
        if not all(c in design_df_local.columns for c in req_cols):
            raise ValueError(
                f"Area data missing one or more required columns: {req_cols}."
            )

        design_df_local["map_area"] = pd.to_numeric(
            design_df_local["map_area"], errors="coerce"
        ).fillna(0)
        design_df_local = design_df_local[
            design_df_local["map_area"] > 1e-9
        ].reset_index(drop=True)
        if design_df_local.empty:
            raise ValueError(
                "No classes found with positive area after cleaning area data."
            )
        if design_df_local["map_code"].duplicated().any():
            raise ValueError(
                "Duplicate map codes found in final area data. Check input or class editing step."
            )

        design_df_indexed_local = design_df_local.set_index("map_code")
        total_map_area_val = design_df_indexed_local["map_area"].sum()
        num_classes_val = len(design_df_indexed_local)
        if not (total_map_area_val > 1e-9):
            raise ValueError("Total map area is effectively zero.")
        logger.debug(
            f"Area data prepared: {num_classes_val} classes, Total Area: {total_map_area_val:.2f}"
        )

        report_to_ui_and_log(
            "Calculating baseline total sample size (N)...", log_level="debug"
        )
        numerator_overall = z_score_val**2 * target_oa_param * (1 - target_oa_param)
        denominator_overall = allowable_err_overall**2
        n_total_formula = (
            ceil(numerator_overall / denominator_overall)
            if numerator_overall > 0 and denominator_overall > 0
            else 0
        )

        n_total_adjusted = max(n_total_formula, num_classes_val * min_n_per_class)
        logger.debug(
            f"Baseline N (formula): {n_total_formula}, N (min adjusted): {n_total_adjusted}"
        )

        report_to_ui_and_log(
            f"Performing initial '{alloc_method_param}' allocation...",
            log_level="debug",
        )
        nj_initial_alloc = pd.Series(
            index=design_df_indexed_local.index, dtype=float
        ).fillna(0.0)
        effective_alloc_method = alloc_method_param

        if effective_alloc_method == "Proportional":
            Wj_proportions = design_df_indexed_local["map_area"] / total_map_area_val
            Wj_proportions = Wj_proportions.fillna(0)
            if Wj_proportions.sum() > 1e-9:
                Wj_proportions = Wj_proportions / Wj_proportions.sum()
                nj_initial_alloc = n_total_adjusted * Wj_proportions
            else:
                effective_alloc_method = "Equal"
                warn_msg_prop = "Proportional allocation failed (sum of area proportions is zero). Falling back to Equal allocation."
                report_to_ui_and_log(
                    warn_msg_prop, ui_level="warning", log_level="warning"
                )

        if effective_alloc_method == "Neyman":
            if not expected_user_accuracy_widgets:
                raise ValueError(
                    "Neyman allocation requires Expected User Accuracy values, but widgets not provided."
                )

            Sj_std_devs = {}
            valid_ua_found_count = 0
            for code_key, widget_item in expected_user_accuracy_widgets.items():
                ua_val = widget_item.v_model
                if 0 < ua_val < 1:
                    Sj_std_devs[code_key] = np.sqrt(ua_val * (1 - ua_val))
                    valid_ua_found_count += 1
                else:
                    Sj_std_devs[code_key] = 0.0
                    logger.warning(
                        f"Expected UA for class code {code_key} is {ua_val:.2f} (outside 0-1 exclusive range or at boundary). Std Dev (Sj) set to 0 for this class in Neyman."
                    )

            if valid_ua_found_count == 0 and num_classes_val > 0:
                logger.warning(
                    "Neyman allocation: No classes have valid Expected UA (0 < UA < 1). Sum(Wk*Sk) will be zero."
                )

            Sj_series = (
                pd.Series(Sj_std_devs).reindex(design_df_indexed_local.index).fillna(0)
            )
            Wj_proportions_neyman = (
                design_df_indexed_local["map_area"] / total_map_area_val
            ).fillna(0)

            if Wj_proportions_neyman.sum() <= 1e-9 and num_classes_val > 0:
                effective_alloc_method = "Equal"
                warn_msg_neyman_area = "Neyman allocation: Sum of area proportions is zero. Falling back to Equal allocation."
                report_to_ui_and_log(
                    warn_msg_neyman_area, ui_level="warning", log_level="warning"
                )
            else:
                WjSj_products = (Wj_proportions_neyman * Sj_series).fillna(0)
                sum_WkSk_val = WjSj_products.sum()

                if sum_WkSk_val <= 1e-9:
                    if Wj_proportions_neyman.sum() > 1e-9:
                        effective_alloc_method = "Proportional"
                        nj_initial_alloc = n_total_adjusted * (
                            Wj_proportions_neyman / Wj_proportions_neyman.sum()
                        )
                        warn_msg_neyman_sumwksk1 = "Neyman allocation: Sum(Wk*Sk) is zero. Falling back to Proportional allocation based on areas."
                        report_to_ui_and_log(
                            warn_msg_neyman_sumwksk1,
                            ui_level="warning",
                            log_level="warning",
                        )
                    else:
                        effective_alloc_method = "Equal"
                        warn_msg_neyman_sumwksk2 = "Neyman allocation: Sum(Wk*Sk) and Sum(Wk) are zero. Falling back to Equal allocation."
                        report_to_ui_and_log(
                            warn_msg_neyman_sumwksk2,
                            ui_level="warning",
                            log_level="warning",
                        )
                else:
                    allocation_ratios_neyman = WjSj_products / sum_WkSk_val
                    nj_initial_alloc = (
                        n_total_adjusted * allocation_ratios_neyman.fillna(0)
                    )

        if effective_alloc_method == "Equal":
            if num_classes_val > 0:
                nj_initial_alloc.loc[:] = n_total_adjusted / num_classes_val
            else:
                nj_initial_alloc.loc[:] = 0.0
                logger.warning("Equal allocation attempted with zero classes.")

        logger.debug(
            f"Initial allocation (float) using {effective_alloc_method}:\n{nj_initial_alloc.to_string()}"
        )

        report_to_ui_and_log(
            "Applying constraints (minimum samples per class, integer rounding)...",
            log_level="debug",
        )
        nj_initial_alloc = nj_initial_alloc.reindex(
            design_df_indexed_local.index
        ).fillna(0)
        nj_final_alloc = (
            nj_initial_alloc.apply(ceil).astype(int).clip(lower=min_n_per_class)
        )

        if (
            objective == "Target Class Precision"
            and target_class_code_val in nj_final_alloc.index
        ):
            if n_req_target_class > nj_final_alloc.loc[target_class_code_val]:
                logger.info(
                    f"Adjusting samples for target class '{target_class_name_display}' (Code: {target_class_code_val}) from {nj_final_alloc.loc[target_class_code_val]} to required {n_req_target_class}."
                )
                nj_final_alloc.loc[target_class_code_val] = n_req_target_class

        n_samples_per_class_final_dict = nj_final_alloc.to_dict()
        logger.debug(
            f"Final sample counts per class (integer, min constrained):\n{nj_final_alloc.to_string()}"
        )

        summary_df_local = design_df_local[
            ["map_code", "map_edited_class", "map_area"]
        ].copy()
        summary_df_local["Initial Allocation (Float)"] = (
            summary_df_local["map_code"].map(nj_initial_alloc).round(2)
        )
        summary_df_local["Final Sample Count"] = (
            summary_df_local["map_code"].map(nj_final_alloc).astype(int)
        )
        summary_df_local.fillna(
            {"Initial Allocation (Float)": 0, "Final Sample Count": 0}, inplace=True
        )
        actual_total_samples_val = summary_df_local["Final Sample Count"].sum()

        summary_output_str = (
            f"--- Final Sample Allocation (Objective: {objective}) ---\n"
        )
        summary_output_str += f"Allocation Method Used: {effective_alloc_method} (Requested: {alloc_method_param})\n"
        summary_output_str += (
            f"Confidence: {confidence_param*100:.1f}%, Z-score: {z_score_val:.3f}\n"
        )
        if objective == "Target Class Precision":
            summary_output_str += f"Target Class: '{target_class_name_display}' (Code: {target_class_code_val}), Required Samples >= {n_req_target_class}\n"
            summary_output_str += f"  Expected UA (Target Class): {target_class_ua_param:.2f}, Allowable Error (Target Class): {target_class_err_param:.3f}\n"
        else:
            summary_output_str += f"Target Overall Accuracy: {target_oa_param:.2f}, Allowable Error (Overall): {allowable_err_overall:.3f}\n"
        summary_output_str += f"Minimum Samples per Class: {min_n_per_class}\n---\n"

        display_cols_summary = [
            "map_edited_class",
            "map_code",
            "map_area",
            "Initial Allocation (Float)",
            "Final Sample Count",
        ]
        summary_output_str += (
            summary_df_local[display_cols_summary].round(2).to_string(index=False)
        )
        summary_output_str += (
            f"\n\nCalculated N (baseline, min adjusted): {n_total_adjusted}\n"
        )
        summary_output_str += (
            f"Final Total Sample Size (after constraints): {actual_total_samples_val}"
        )

        results_dict["summary_text"] = summary_output_str
        report_to_ui_and_log(summary_output_str, log_level="info")

        report_to_ui_and_log("Generating sample points...", log_level="info")
        generated_samples_df_local = generate_sample_points(
            map_file_path_str_arg,
            final_area_df_arg[["map_code", "map_edited_class"]].copy(),
            n_samples_per_class_final_dict,
        )

        if (
            generated_samples_df_local is not None
            and not generated_samples_df_local.empty
        ):
            sample_points_df = generated_samples_df_local
            report_to_ui_and_log(
                f"Successfully generated {len(generated_samples_df_local)} sample points.",
                log_level="info",
            )
        else:
            sample_points_df = pd.DataFrame()
            report_to_ui_and_log(
                "Sample point generation failed or produced no points. Check logs for details.",
                ui_level="warning",
                log_level="warning",
            )

        results_dict["generated_samples_df"] = sample_points_df

        if not sample_points_df.empty:
            report_to_ui_and_log(
                "Outputting sample points to files...", log_level="info"
            )
            output_sample_points(sample_points_df, output_dir_str_arg)

        final_report_msg = f"\nSample design and point generation complete.\nOutput files saved in directory:\n -> {output_dir_str_arg}\n"
        final_report_msg += "Click 'Show/Update Samples on Map' in Step 3 to visualize if points were generated."
        report_to_ui_and_log(final_report_msg, log_level="info")
        report_to_ui_and_log(
            {"msg": "--- Sample Design Run Complete ---", "type": "success"},
            ui_level="summary_end",
        )

        results_dict["success"] = True
        results_dict["actual_total_samples"] = actual_total_samples_val
        results_dict["summary_df"] = summary_df_local

    except (ValueError, AssertionError) as ve:
        err_msg = f"ERROR during sample design validation or parameter setup: {ve}"
        logger.error(err_msg)
        report_to_ui_and_log(err_msg, ui_level="error", log_level="error")
        report_to_ui_and_log(
            {
                "msg": f"--- Sample Design Failed ---\n{ve}\n\nPlease check parameters and logs, then try again.",
                "type": "error",
            },
            ui_level="summary_end",
        )
        results_dict["error_message"] = str(ve)
    except ImportError as ie:
        err_msg = f"IMPORT ERROR during sample design: {ie}. A required library might be missing."
        logger.critical(err_msg)
        report_to_ui_and_log(err_msg, ui_level="error", log_level="critical")
        report_to_ui_and_log(
            {
                "msg": f"--- Sample Design Failed (Import Error) ---\n{ie}",
                "type": "error",
            },
            ui_level="summary_end",
        )
        results_dict["error_message"] = str(ie)
    except Exception as e:
        err_msg = f"An unexpected error occurred during sample design: {e}"
        logger.exception("Unexpected error in run_sample_design:")
        report_to_ui_and_log(err_msg, ui_level="error", log_level="error")
        report_to_ui_and_log(
            {
                "msg": f"--- Unexpected Error ---\n{e}\n\nPlease report this issue. Check logs for details.",
                "type": "error",
            },
            ui_level="summary_end",
        )
        results_dict["error_message"] = str(e)

    logger.info(">>> Exiting run_sample_design function.")
    return results_dict
