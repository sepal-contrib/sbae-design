# sampling_design_tool/processing.py

import os
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import sys
import random
from math import ceil
import ipywidgets as widgets # For type hinting sample_size_output

# Optional dependencies
try:
    import rasterio
    import rasterio.warp
    _rasterio_available = True
except ImportError:
    print("WARNING (processing): rasterio not found.")
    _rasterio_available = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    _geopandas_available = True
except ImportError:
    print("WARNING (processing): geopandas/shapely not found.")
    _geopandas_available = False

try:
    from pyproj import CRS, Transformer
    _pyproj_available = True
    print("DEBUG SUCCESS (processing): pyproj imported successfully.") # Add this
except ImportError as e:
    print(f"CRITICAL WARNING (processing): FAILED to import pyproj. Error: {e}") # Make it stand out
    _pyproj_available = False

try:
    from scipy.special import ndtri
    _scipy_available = True
except ImportError:
    print("Warning (processing): scipy not found. Z-score limited.")
    _scipy_available = False

# --- Constants ---
target_crs_epsg = 4326 # WGS 84
target_crs_str = f"EPSG:{target_crs_epsg}"


# --- Globals used by run_sample_design ---
# These will be set by AppController before calling run_sample_design
sample_size_output: widgets.Output = None # Or sui.Alert
sample_points_df: pd.DataFrame = None
objective_widget = None
target_class_widget = None
target_class_allowable_error_widget = None
expected_user_accuracy_widgets = None # Dict of sliders
allocation_method_widget = None
target_overall_accuracy_widget = None
allowable_error_widget = None
confidence_level_widget = None
min_samples_per_class_widget = None
# --- End Globals ---


def get_z_score(confidence_level):
    if confidence_level == 0.90: return 1.645
    if confidence_level == 0.95: return 1.96
    if confidence_level == 0.99: return 2.58
    if _scipy_available:
        try:
            p = (1 + confidence_level) / 2.0
            return ndtri(p) if 0 < p < 1 else (_ for _ in ()).throw(ValueError(f"Invalid probability {p}"))
        except Exception as e:
            print(f"Error calculating Z-score: {e}")
            raise ValueError(f"Could not calculate Z-score for {confidence_level}.")
    else:
        raise ValueError(f"Accurate Z-score for {confidence_level} requires scipy, or use 0.90, 0.95, 0.99.")

def get_output_dir(input_file_path):
    print(f"DEBUG: get_output_dir called for {input_file_path}")
    if input_file_path and isinstance(input_file_path, (str, Path)) and Path(input_file_path).is_file():
        try:
            p = Path(input_file_path)
            dirname = p.parent
            basename_no_ext = p.stem
            safe_basename = "".join(c if c.isalnum() else "_" for c in basename_no_ext)
            subdir = f'sae_design_{safe_basename}'
            output_dir = dirname / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Output directory: {output_dir}")
            return str(output_dir)
        except Exception as e:
            print(f"Error creating output directory for {input_file_path}: {e}") # Clarified error
            return None
    else:
        print(f"Error: Invalid input file path: {input_file_path}")
        return None

def compute_map_area(input_map_file_path):
    print(f"DEBUG: compute_map_area for {input_map_file_path}.")
    area_df = None
    if not input_map_file_path or not isinstance(input_map_file_path, (str, Path)) or not Path(input_map_file_path).exists():
        print("Error: Invalid file path.")
        return None

    file_path = Path(input_map_file_path)
    basename = file_path.name
    raster_extensions = ('.tif', '.tiff', '.img', '.pix', '.rst', '.grd', '.vrt', '.hdf', '.h5', '.jpeg2000')
    vector_extensions = ('.shp', '.sqlite', '.gdb', '.geojson', '.json', '.gml', '.kml', '.tab', '.mif')

    if file_path.suffix.lower() in raster_extensions:
        if not _rasterio_available:
            print("ERROR: rasterio not available for raster processing!")
            return None
        print(f"Processing {basename} as Raster...")
        try:
            with rasterio.open(file_path) as src:
                crs = src.crs
                print(f"Info: Raster CRS: {crs}")
                if crs and crs.is_geographic: print(f"Warning: CRS geographic. Area in sq. degrees.")
                elif not crs: print("Warning: No CRS defined.")

                if not src.res or len(src.res) < 2 or not all(src.res):
                    print(f"Error: Invalid resolution: {src.res}.")
                    return None

                pixel_area = abs(src.res[0] * src.res[1])
                data = src.read(1)
                nodata = src.nodata
                unique_values, counts = np.unique(data, return_counts=True)
                
                valid_mask = np.ones_like(unique_values, dtype=bool)
                if nodata is not None:
                    valid_mask &= (~np.isnan(unique_values) if np.isnan(nodata) else (unique_values != nodata))
                valid_mask &= ~np.isnan(unique_values)

                unique_values, counts = unique_values[valid_mask], counts[valid_mask]
                if unique_values.size == 0:
                    print("No valid class values found.")
                    return None
                
                area_df = pd.DataFrame({'map_code': unique_values, 'map_area': counts * pixel_area})
                area_df = area_df[area_df['map_area'] > 0].reset_index(drop=True)
                if area_df.empty:
                    print("No classes with positive area.")
                    return None
                print(f"Computed areas for {len(area_df)} raster classes.")
        except Exception as e:
            print(f"!!! Error reading/processing raster {basename}: {e}")
            traceback.print_exc()
            return None

    elif file_path.suffix.lower() in vector_extensions:
        if not _geopandas_available:
            print("ERROR: geopandas not available for vector processing!")
            return None
        print(f"Processing {basename} as Vector...")
        try:
            gdf = gpd.read_file(file_path)
            if gdf.empty:
                print("Vector file is empty.")
                return None

            geom_col = gdf.geometry.name
            assert geom_col in gdf.columns and not gdf[geom_col].isnull().all() and not gdf[geom_col].is_empty.all(), "No valid geometries."
            
            crs = gdf.crs
            print(f"Info: Vector CRS: {crs}")
            if crs and crs.is_geographic: print("Warning: CRS geographic. Area in sq. degrees.")
            elif not crs: print("Warning: No CRS defined.")

            attribute_cols = [col for col in gdf.columns if col != geom_col]
            assert attribute_cols, "No attribute columns."
            map_code_col = attribute_cols[0] # Use first attribute as class identifier
            print(f"Using '{map_code_col}' as map code.")
            assert map_code_col in gdf.columns and not gdf[map_code_col].isnull().all(), f"Map code column '{map_code_col}' invalid."

            try:
                gdf[geom_col] = gdf[geom_col].buffer(0) # Fix invalid geometries
                gdf = gdf[gdf[geom_col].is_valid & ~gdf[geom_col].is_empty]
                assert not gdf.empty, "No valid geometries after cleaning."
                gdf['map_area'] = gdf.geometry.area
                gdf['map_area'] = gdf['map_area'].replace([np.inf, -np.inf], np.nan).fillna(0)
                gdf = gdf[(gdf['map_area'] > 0) & (gdf[map_code_col].notna())].reset_index(drop=True)
            except Exception as area_error:
                print(f"!!! Error calculating geometry area: {area_error}")
                traceback.print_exc()
                return None
            
            assert not gdf.empty, "No valid features with positive area/map code."
            
            area_df = gdf.groupby(map_code_col)['map_area'].sum().reset_index()
            area_df.columns = ['map_code', 'map_area']
            area_df = area_df[area_df['map_area'] > 0].reset_index(drop=True)
            assert not area_df.empty, "No classes with positive total area."
            print(f"Computed areas for {len(area_df)} vector classes.")

        except Exception as e:
            print(f"!!! Error reading/processing vector {basename}: {e}")
            traceback.print_exc()
            return None
    else:
        print(f"Unsupported file format: {basename}")
        return None

    if area_df is not None and 'map_edited_class' not in area_df.columns:
        area_df['map_edited_class'] = area_df['map_code'].astype(str)
    return area_df


def generate_sample_points(map_file, final_area_df_col_indexed, n_samples_per_class):
    print("--- Running generate_sample_points ---")
    global target_crs_epsg, target_crs_str # Use constants defined in this module
    sample_points_list = []
    if not _pyproj_available:
        print("Error: pyproj required for sample point generation.")
        return pd.DataFrame()

    if final_area_df_col_indexed is None or not all(c in final_area_df_col_indexed.columns for c in ['map_code', 'map_edited_class']):
        print("Error: final_area_df for sample generation is invalid.")
        return pd.DataFrame()

    class_lookup = final_area_df_col_indexed.set_index('map_code')['map_edited_class'].to_dict()
    map_file_path = Path(map_file)
    basename = map_file_path.name
    raster_extensions = ('.tif', '.tiff', '.img', '.pix', '.rst', '.grd', '.vrt', '.hdf', '.h5', '.jpeg2000')
    vector_extensions = ('.shp', '.sqlite', '.gdb', '.geojson', '.json', '.gml', '.kml', '.tab', '.mif')

    try:
        if map_file_path.suffix.lower() in raster_extensions:
            if not _rasterio_available:
                print("ERROR: rasterio needed for raster sample generation!")
                return pd.DataFrame()
            print(f"Generating samples from Raster: {basename}")
            with rasterio.open(map_file_path) as src:
                transform_affine = src.transform
                data = src.read(1)
                source_crs = src.crs
                print(f"Raster source CRS: {source_crs}")
                transformer_proj = None
                if source_crs and _pyproj_available and CRS.from_user_input(source_crs).to_epsg() != target_crs_epsg:
                    try:
                        transformer_proj = Transformer.from_crs(source_crs, f"EPSG:{target_crs_epsg}", always_xy=True)
                        print(f"Reprojection transformer created for raster.")
                    except Exception as e:
                        print(f"Warning: Failed to create transformer for raster: {e}.")
                elif not source_crs: print("Warning: No CRS for raster. Assuming EPSG:4326.")
                
                for map_code, n_samples in n_samples_per_class.items():
                    if n_samples <= 0: continue
                    edited_class = class_lookup.get(map_code, f"Code {map_code}")
                    edited_class_str = str(edited_class) if pd.notna(edited_class) else f"Code {map_code}"
                    rows, cols = np.where(data == map_code)
                    num_available_pixels = len(rows)
                    
                    if num_available_pixels == 0:
                        print(f"Warning: No pixels for {map_code} ({edited_class_str}).")
                        continue

                    num_samples_to_take = min(n_samples, num_available_pixels)
                    if num_samples_to_take < n_samples:
                        print(f"Warning: Requested {n_samples} for {edited_class_str}, only {num_available_pixels} available.")
                    
                    sampled_indices = random.sample(range(num_available_pixels), num_samples_to_take)
                    sampled_rows, sampled_cols = rows[sampled_indices], cols[sampled_indices]

                    for r_idx, c_idx in zip(sampled_rows, sampled_cols):
                        x_coord, y_coord = transform_affine * (c_idx + 0.5, r_idx + 0.5) # Center of pixel
                        lon, lat = x_coord, y_coord
                        if transformer_proj:
                            try: lon, lat = transformer_proj.transform(x_coord, y_coord)
                            except Exception as proj_error:
                                print(f"Warning: Reproject failed for raster point ({x_coord},{y_coord}): {proj_error}. Skipping.")
                                continue
                        sample_points_list.append({'latitude': lat, 'longitude': lon, 'map_code': map_code, 'map_edited_class': edited_class_str})
                    print(f"Generated {num_samples_to_take} points for class '{edited_class_str}' (Raster)")
                del data

        elif map_file_path.suffix.lower() in vector_extensions:
            if not _geopandas_available:
                print("ERROR: geopandas needed for vector sample generation!")
                return pd.DataFrame()
            print(f"Generating samples from Vector: {basename}")
            gdf = gpd.read_file(map_file_path)
            if gdf.empty: raise ValueError("Input vector file empty for sample generation.")
            
            source_crs = gdf.crs
            print(f"Vector source CRS: {source_crs}")
            geom_col = gdf.geometry.name
            attribute_cols = [col for col in gdf.columns if col != geom_col]
            map_code_col = attribute_cols[0] if attribute_cols else None # Assuming first attr is class
            assert map_code_col and map_code_col in gdf.columns, f"Map code column '{map_code_col}' not valid for vector."

            transformer_proj = None
            if source_crs and _pyproj_available and CRS.from_user_input(source_crs).to_epsg() != target_crs_epsg:
                try:
                    transformer_proj = Transformer.from_crs(source_crs, f"EPSG:{target_crs_epsg}", always_xy=True)
                    print(f"Reprojection transformer created for vector.")
                except Exception as e:
                    print(f"Warning: Failed to create transformer for vector: {e}.")
            elif not source_crs: print("Warning: No CRS for vector. Assuming EPSG:4326.")

            for map_code, n_samples in n_samples_per_class.items():
                if n_samples <= 0: continue
                edited_class = class_lookup.get(map_code, f"Code {map_code}")
                edited_class_str = str(edited_class) if pd.notna(edited_class) else f"Code {map_code}"
                class_gdf = gdf[gdf[map_code_col] == map_code].copy()
                class_gdf[geom_col] = class_gdf[geom_col].buffer(0)
                class_gdf = class_gdf[class_gdf[geom_col].is_valid & ~class_gdf[geom_col].is_empty]

                if class_gdf.empty:
                    print(f"Warning: No valid/non-empty geometries for {map_code} ({edited_class_str}) in vector.")
                    continue

                total_bounds = class_gdf.total_bounds
                if not (np.all(np.isfinite(total_bounds)) and len(total_bounds) == 4):
                    print(f"Warning: Invalid bounds {total_bounds} for {edited_class_str} (vector). Skipping.")
                    continue
                minx, miny, maxx, maxy = total_bounds
                if not (minx <= maxx and miny <= maxy):
                     print(f"Warning: Illogical bounds [{minx},{miny},{maxx},{maxy}] for {edited_class_str} (vector). Skipping.")
                     continue
                
                samples_generated = 0
                max_attempts = n_samples * 100 
                print(f"Attempting {n_samples} points for class '{edited_class_str}' (Vector)...")

                for _ in range(max_attempts):
                    if samples_generated >= n_samples: break
                    rand_x = random.uniform(minx, maxx)
                    rand_y = random.uniform(miny, maxy)
                    point = Point(rand_x, rand_y)

                    if class_gdf.contains(point).any():
                        lon, lat = rand_x, rand_y
                        if transformer_proj:
                            try: lon, lat = transformer_proj.transform(rand_x, rand_y)
                            except Exception as proj_error:
                                print(f"Warning: Reproject failed for vector point ({rand_x},{rand_y}): {proj_error}. Skipping.")
                                continue
                        sample_points_list.append({'latitude': lat, 'longitude': lon, 'map_code': map_code, 'map_edited_class': edited_class_str})
                        samples_generated += 1
                
                print(f"Generated {samples_generated}/{n_samples} points for class '{edited_class_str}' (Vector).")
                if samples_generated < n_samples:
                    print(f"Warning: Low generation rate for {edited_class_str} (Vector) after {max_attempts} attempts.")
            del gdf, class_gdf
        else:
            print(f"Unsupported format for sample generation: {basename}")
    except Exception as e:
        print(f"!!! Error during sample point generation: {e}")
        traceback.print_exc()

    print(f"--- generate_sample_points finished. Generated {len(sample_points_list)} points total. ---")
    return pd.DataFrame(sample_points_list) if sample_points_list else None


def output_sample_points(df_samples_to_output: pd.DataFrame, output_dir_path: str):
    """Outputs generated sample points to CSV and GeoJSON files."""
    # Access the global sample_size_output (assumed to be an Alert or Output widget)
    # This is a bit fragile; ideally, a logger or callback would be passed.
    global sample_size_output 
    output_widget_ref = sample_size_output

    def print_msg(msg):
        if output_widget_ref:
            if hasattr(output_widget_ref, 'add_msg'): output_widget_ref.add_msg(str(msg) + "\n") # for sui.Alert
            elif isinstance(output_widget_ref, widgets.Output): # for basic Output
                with output_widget_ref: print(msg)
        else: print(msg)

    print_msg("--- Outputting Sample Points ---")
    if df_samples_to_output is None or df_samples_to_output.empty:
        print_msg("No sample points to save.")
        return
    if not output_dir_path or not os.path.isdir(output_dir_path):
        print_msg(f"Error: Output directory '{output_dir_path}' invalid for saving samples.")
        return

    csv_filepath = os.path.join(output_dir_path, "sample_points.csv")
    try:
        df_samples_to_output.to_csv(csv_filepath, index=False)
        print_msg(f"Saved: {csv_filepath}")
    except Exception as e:
        print_msg(f"!!! Error saving CSV: {e}")
        traceback.print_exc()

    geojson_filepath = os.path.join(output_dir_path, "sample_points.geojson")
    try:
        if not _geopandas_available:
            print_msg("GeoJSON output requires geopandas.")
            return
        if not all(c in df_samples_to_output.columns for c in ['latitude', 'longitude']):
            print_msg("Error: Missing lat/lon for GeoJSON output.")
            return
        
        df_samples_to_output['latitude'] = pd.to_numeric(df_samples_to_output['latitude'], errors='coerce')
        df_samples_to_output['longitude'] = pd.to_numeric(df_samples_to_output['longitude'], errors='coerce')
        valid_points_gdf = df_samples_to_output.dropna(subset=['latitude', 'longitude'])
        
        if valid_points_gdf.empty:
            print_msg("Error: No valid coordinates found for GeoJSON output after cleaning.")
            return

        gdf_samples_out = gpd.GeoDataFrame(
            valid_points_gdf,
            geometry=gpd.points_from_xy(valid_points_gdf.longitude, valid_points_gdf.latitude),
            crs=f"EPSG:{target_crs_epsg}"
        )
        gdf_samples_out.to_file(geojson_filepath, driver='GeoJSON')
        print_msg(f"Saved: {geojson_filepath}")
    except Exception as e:
        print_msg(f"!!! Error saving GeoJSON: {e}")
        traceback.print_exc()


def run_sample_design(map_file_path_str: str, final_area_df_input: pd.DataFrame, output_dir_str: str):
    """
    Calculates sample sizes, allocates samples, generates points, and outputs results.
    Relies on global widget variables being set by the AppController.
    """
    # Access global variables set by AppController
    global sample_points_df, sample_size_output
    global objective_widget, target_class_widget, target_class_allowable_error_widget
    global expected_user_accuracy_widgets, allocation_method_widget
    global target_overall_accuracy_widget, allowable_error_widget
    global confidence_level_widget, min_samples_per_class_widget

    # Reset global result
    sample_points_df = None 
    print(">>> Entered run_sample_design function (processing.py).")
    output_widget_ref = sample_size_output # Use the global reference

    def report_summary(summary_str, msg_type='info'):
        if output_widget_ref:
            if hasattr(output_widget_ref, 'reset') and hasattr(output_widget_ref, 'add_msg') and hasattr(output_widget_ref, 'type'): # sui.Alert
                output_widget_ref.reset()
                output_widget_ref.type = msg_type
                output_widget_ref.add_msg(str(summary_str))
                output_widget_ref.show()
            elif isinstance(output_widget_ref, widgets.Output): # widgets.Output
                with output_widget_ref:
                    from IPython.display import clear_output # Local import
                    clear_output(wait=True)
                    print(f"[{msg_type.upper()}]\n{summary_str}")
        else: print(summary_str)

    def report_progress(msg):
        if output_widget_ref:
            if hasattr(output_widget_ref, 'add_msg'): # sui.Alert
                output_widget_ref.add_msg(str(msg) + "\n")
                output_widget_ref.show()
            elif isinstance(output_widget_ref, widgets.Output): # widgets.Output
                with output_widget_ref: print(msg)
        else: print(msg)

    report_summary("--- Running Sample Design Calculation ---", msg_type='info')
    try:
        report_progress("Retrieving parameters...")
        objective = objective_widget.v_model
        target_oa = target_overall_accuracy_widget.v_model
        allowable_err = allowable_error_widget.v_model
        confidence = confidence_level_widget.v_model
        min_n_class = min_samples_per_class_widget.v_model
        alloc_method = allocation_method_widget.v_model
        z = get_z_score(confidence)
        
        assert 0 < target_oa < 1, "Target OA must be between 0 and 1."
        assert 0 < allowable_err < 1, "Allowable Error (Overall) must be between 0 and 1."
        assert 0.8 <= confidence < 1, "Confidence Level must be between 0.8 and 1 (exclusive)."
        assert min_n_class > 0, "Min Samples per Class must be positive."

        target_class_code, n_target_class_required, target_class_display_name = None, 0, "N/A"
        if objective == 'Target Class Precision':
            target_class_code = target_class_widget.v_model
            if target_class_code is None:
                raise ValueError("Please select the Target Class for 'Target Class Precision' objective.")
            target_class_display_name = next((opt['text'] for opt in target_class_widget.items if opt['value'] == target_class_code), f"Class {target_class_code}")
            target_class_error = target_class_allowable_error_widget.v_model
            assert 0 < target_class_error < 1, "Target Class Allowable Error must be between 0 and 1."
            assert expected_user_accuracy_widgets and target_class_code in expected_user_accuracy_widgets, f"Expected UA widget missing for target class {target_class_code}."
            target_class_ua = expected_user_accuracy_widgets[target_class_code].v_model
            assert 0 < target_class_ua < 1, f"Expected UA for '{target_class_display_name}' (Code {target_class_code}) must be between 0 and 1."
            
            numerator_tc = z**2 * target_class_ua * (1 - target_class_ua)
            denominator_tc = target_class_error**2
            n_target_class_required = ceil(numerator_tc / denominator_tc) if numerator_tc > 0 and denominator_tc > 0 else 0
            n_target_class_required = max(n_target_class_required, min_n_class)
        
        report_progress("Preparing area data...")
        assert final_area_df_input is not None and not final_area_df_input.empty, "Final class area data is missing or empty."
        design_df = final_area_df_input.copy()
        required_cols = ['map_code', 'map_area', 'map_edited_class']
        assert all(c in design_df.columns for c in required_cols), f"Area data missing one or more required columns: {required_cols}."
        design_df['map_area'] = pd.to_numeric(design_df['map_area'], errors='coerce').fillna(0)
        design_df = design_df[design_df['map_area'] > 0].reset_index(drop=True)
        assert not design_df.empty, "No classes found with positive area after cleaning."
        assert not design_df['map_code'].duplicated().any(), "Duplicate map codes found in final area data."
        
        design_df_indexed = design_df.set_index('map_code')
        total_map_area = design_df_indexed['map_area'].sum()
        num_classes = len(design_df_indexed)
        assert total_map_area > 1e-9, "Total map area is effectively zero."

        report_progress("Calculating baseline sample size...")
        numerator = z**2 * target_oa * (1 - target_oa)
        denominator = allowable_err**2
        n_target_formula = ceil(numerator / denominator) if numerator > 0 and denominator > 0 else 0
        n_target_min_adjusted = max(n_target_formula, num_classes * min_n_class)

        report_progress(f"Performing initial '{alloc_method}' allocation...")
        nj_initial = pd.Series(index=design_df_indexed.index, dtype=float).fillna(0.0)
        current_alloc_method = alloc_method

        if current_alloc_method == 'Proportional':
            Wj = design_df_indexed['map_area'] / total_map_area
            Wj = Wj.fillna(0)
            if Wj.sum() > 1e-9: Wj = Wj / Wj.sum(); nj_initial = (n_target_min_adjusted * Wj)
            else: current_alloc_method = 'Equal'; report_progress("WARN: Proportional allocation failed. Falling back to Equal.")
        
        if current_alloc_method == 'Neyman': # Could be after fallback from Proportional
            assert expected_user_accuracy_widgets, "Neyman allocation requires Expected User Accuracy widgets."
            Sj_values = {}
            valid_ua_count = 0
            for code, widget in expected_user_accuracy_widgets.items():
                ua = widget.v_model
                if 0 < ua < 1: Sj_values[code] = np.sqrt(ua * (1 - ua)); valid_ua_count +=1
                else: Sj_values[code] = 0.0; report_progress(f"Warning: Expected UA for class {code} is {ua}. Setting std dev to 0.")
            Sj = pd.Series(Sj_values).reindex(design_df_indexed.index).fillna(0)
            assert valid_ua_count > 0, "Neyman requires at least one class with valid Expected UA (0 < UA < 1)."
            
            Wj = design_df_indexed['map_area'] / total_map_area
            Wj = Wj.fillna(0)
            if Wj.sum() > 1e-9: Wj = Wj / Wj.sum()
            else: Wj = pd.Series(0.0, index=design_df_indexed.index); current_alloc_method = 'Equal'; report_progress("WARN: Total map area zero for Neyman. Falling back to Equal.")

            if current_alloc_method == 'Neyman': # Check again if still Neyman
                WjSj = (Wj * Sj).fillna(0)
                sum_WkSk = WjSj.sum()
                if sum_WkSk <= 1e-9:
                    if Wj.sum() > 1e-9: nj_initial = (n_target_min_adjusted * Wj); current_alloc_method = 'Proportional'; report_progress("WARN: Neyman Sum(Wk*Sk) zero. Falling back to Proportional.")
                    else: current_alloc_method = 'Equal'; report_progress("WARN: Neyman Sum(Wk*Sk) zero and Wj zero. Falling back to Equal.")
                else:
                    allocation_ratios = WjSj / sum_WkSk
                    nj_initial = (n_target_min_adjusted * allocation_ratios.fillna(0))

        if current_alloc_method == 'Equal': # Handles explicit Equal or fallbacks
            nj_initial.loc[:] = n_target_min_adjusted / num_classes if num_classes > 0 else 0.0
        
        report_progress("Applying constraints (minimum samples per class)...")
        nj_initial = nj_initial.reindex(design_df_indexed.index).fillna(0)
        nj_final = nj_initial.apply(ceil).astype(int).clip(lower=min_n_class)
        if objective == 'Target Class Precision' and target_class_code in nj_final.index:
            if n_target_class_required > nj_final.loc[target_class_code]:
                nj_final.loc[target_class_code] = n_target_class_required
        n_samples_per_class_final = nj_final.to_dict()

        summary_df = design_df[['map_code', 'map_edited_class', 'map_area']].copy()
        summary_df['Initial Allocation (Float)'] = summary_df['map_code'].map(nj_initial).round(2)
        summary_df['Final Sample Count'] = summary_df['map_code'].map(nj_final).astype(int)
        summary_df.fillna({'Initial Allocation (Float)': 0, 'Final Sample Count': 0}, inplace=True)
        actual_total_samples = summary_df['Final Sample Count'].sum()

        summary_output_str = f"--- Final Sample Allocation (Objective: {objective}) ---\n"
        summary_output_str += f"Allocation Method: {alloc_method} (Applied as: {current_alloc_method})\n"
        summary_output_str += f"Confidence: {confidence*100:.1f}%, Z: {z:.3f}\n"
        if objective == 'Target Class Precision':
            summary_output_str += f"Target Class: '{target_class_display_name}' ({target_class_code}), Required >= {n_target_class_required}\n"
            summary_output_str += f"Target UA: {target_class_ua:.2f}, Allowable Error (Class): {target_class_error:.3f}\n"
        else:
            summary_output_str += f"Target OA: {target_oa:.2f}, Allowable Error (Overall): {allowable_err:.3f}\n"
        summary_output_str += f"Minimum Samples/Class: {min_n_class}\n---\n"
        display_cols = ['map_edited_class', 'map_code', 'map_area', 'Initial Allocation (Float)', 'Final Sample Count']
        summary_output_str += summary_df[display_cols].round(2).to_string(index=False)
        summary_output_str += f"\n\nFinal Total Sample Size: {actual_total_samples}"
        report_progress("\n" + summary_output_str + "\n")

        report_progress("Generating sample points...")
        generated_samples = generate_sample_points(
            map_file_path_str,
            final_area_df_input[['map_code', 'map_edited_class']].copy(),
            n_samples_per_class_final
        )

        if generated_samples is not None and not generated_samples.empty:
            sample_points_df = generated_samples # Set global for AppController to pick up
            report_progress(f"Generated {len(generated_samples)} sample points.")
        else:
            sample_points_df = None
            report_progress("Warning: Sample point generation failed or produced no points.")

        if sample_points_df is not None:
            report_progress("Outputting sample points...")
            output_sample_points(sample_points_df, output_dir_str)
        
        report_progress("\nSample design and point generation complete.")
        report_progress(f"Output files saved in directory:\n -> {output_dir_str}")
        report_progress("Click 'Show/Update Samples on Map' in Step 3 to visualize.")
        report_summary("--- Sample Design Run Complete ---", msg_type='success')

    except (ValueError, NameError, KeyError, AttributeError, AssertionError, ImportError) as e:
        err_msg = f"\nERROR during sample design: {e}"
        print(err_msg, file=sys.stderr); traceback.print_exc()
        report_progress(err_msg)
        report_summary(f"--- Sample Design Failed ---\n{e}\n\nPlease check parameters and try again.", msg_type='error')
    except Exception as e:
        err_msg = f"\n!!! An unexpected error occurred: {e}"
        print(err_msg, file=sys.stderr); traceback.print_exc()
        report_progress(err_msg)
        report_summary(f"--- Unexpected Error ---\n{e}\n\nPlease report this issue. Check progress log above.", msg_type='error')

    print(">>> Exiting run_sample_design function (processing.py).")