# sampling_design_tool/map_utils.py

import os
import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
import traceback
import ipyleaflet # Added for type hint

# Optional dependencies - check availability
try:
    import rasterio
    import rasterio.warp # Needed for bounds transform / reprojection
    import rasterio.enums
    _rasterio_available = True
    print("DEBUG SUCCESS (map_utils): rasterio and rasterio.warp imported successfully.") # Add this
except ImportError as e:
    print(f"CRITICAL WARNING (map_utils): FAILED to import rasterio or rasterio.warp. Error: {e}") # Make it stand out
    _rasterio_available = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
    _matplotlib_available = True
except ImportError:
    print("WARNING (map_utils): matplotlib not found.")
    _matplotlib_available = False

try:
    from PIL import Image
    _pillow_available = True
except ImportError:
    print("Warning (map_utils): Pillow not found. Raster display may rely on matplotlib.")
    _pillow_available = False # Set based on actual usage if any, though not directly used here if plt.imsave is primary

# --- Constants ---
target_crs_epsg = 4326
target_crs_str = f"EPSG:{target_crs_epsg}"

# --- Utility Functions ---
def generate_class_color_map(unique_classes):
    """Generates a unique hex color for each class using a colormap."""
    if not _matplotlib_available:
        print("Warning: matplotlib needed for generate_class_color_map")
        return {c:'#808080' for c in unique_classes if pd.notna(c)}
    unique_classes = sorted([c for c in unique_classes if pd.notna(c)])
    n_classes = len(unique_classes)
    if n_classes == 0:
        return {}

    if n_classes <= 10: cmap_name = 'tab10'
    elif n_classes <= 12: cmap_name = 'Set3'
    elif n_classes <= 20: cmap_name = 'tab20'
    else: cmap_name = 'viridis'

    try:
        cmap = plt.colormaps[cmap_name]
        colors = [matplotlib.colors.to_hex(cmap(i / max(1, n_classes - 1))) for i in range(n_classes)]
    except Exception as e:
        print(f"Warning: Colormap '{cmap_name}' failed ({e}). Falling back to viridis.")
        try:
            cmap = plt.colormaps['viridis']
            colors = [matplotlib.colors.to_hex(cmap(i / max(1, n_classes - 1))) for i in range(n_classes)]
        except Exception as fallback_e:
            print(f"ERROR: Fallback colormap failed: {fallback_e}. Returning gray.")
            return {cls: '#808080' for cls in unique_classes}

    return {cls: colors[i] for i, cls in enumerate(unique_classes)}


# --- Helper Function for Map Overlay (_add_overlay_layer) ---
def _add_overlay_layer(map_widget: ipyleaflet.Map, map_file_path: str, legend_data: pd.DataFrame, overlay_bounds, vector_gdf):
    """
    Adds Raster (ImageOverlay/DataURL) or Vector (GeoJSON) layer.
    Returns the created layer object or None if unsuccessful.
    """
    print("DEBUG: (map_utils._add_overlay_layer) Attempting to add overlay layer...")
    
    if not isinstance(map_widget, ipyleaflet.Map):
        print("Error (map_utils._add_overlay_layer): Invalid map widget provided.")
        return None
    if not map_file_path or not os.path.exists(map_file_path):
        print("Error (map_utils._add_overlay_layer): Invalid map file path provided.")
        return None

    created_layer = None
    try:
        map_file_str = str(map_file_path)
        basename = os.path.basename(map_file_str)
        raster_extensions = ('.tif', '.tiff', '.img', '.pix', '.rst', '.grd', '.vrt', '.hdf', '.h5', '.jpeg2000')
        vector_extensions = ('.shp', '.sqlite', '.gdb', '.geojson', '.json', '.gml', '.kml', '.tab', '.mif')

        if map_file_str.lower().endswith(raster_extensions):
            if not _rasterio_available:
                print("Error (map_utils._add_overlay_layer): Rasterio library needed for raster overlay.")
                return None
            if not _matplotlib_available:
                print("Error (map_utils._add_overlay_layer): Matplotlib library needed for raster coloring.")
                return None

            if overlay_bounds and len(overlay_bounds) == 2 and len(overlay_bounds[0]) == 2 and len(overlay_bounds[1]) == 2:
                print(f"DEBUG: (map_utils._add_overlay_layer) Processing Raster {basename} (ImageOverlay/DataURL)...")
                src_raster = None; data_raster = None; image_url = None # Renamed to avoid conflict
                try:
                    with rasterio.open(map_file_str) as src_raster:
                        max_dim = 1000
                        h, w = src_raster.height, src_raster.width
                        if h > max_dim or w > max_dim:
                            scale = max_dim / max(h, w)
                            out_h, out_w = int(h * scale), int(w * scale)
                            print(f"DEBUG: (map_utils._add_overlay_layer) Downsampling {h}x{w} raster to {out_h}x{out_w} for display.")
                            data_raster = src_raster.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest)
                        else:
                            data_raster = src_raster.read(1)
                            print("DEBUG: (map_utils._add_overlay_layer) Using full resolution for display.")
                        nodata_val = src_raster.nodata

                    if legend_data is not None and 'map_code' in legend_data.columns:
                        valid_codes = legend_data['map_code'].dropna().unique()
                        raster_codes_present = np.unique(data_raster[~np.isnan(data_raster)])
                        common_codes = np.intersect1d(valid_codes, raster_codes_present, assume_unique=True)

                        if len(common_codes) > 0:
                            print(f"DEBUG: (map_utils._add_overlay_layer) Colorizing raster based on {len(common_codes)} common codes found in legend.")
                            color_map_hex = generate_class_color_map(common_codes)

                            def hex_to_rgba(hex_color, alpha=0.7):
                                try:
                                    rgb_color = matplotlib.colors.to_rgb(hex_color) # Renamed variable
                                    return tuple(int(c*255) for c in rgb_color) + (int(alpha*255),)
                                except ValueError:
                                    print(f"Warning: Invalid hex color '{hex_color}'. Using transparent gray.")
                                    return (128, 128, 128, 0)

                            color_map_rgba = {code: hex_to_rgba(hex_c) for code, hex_c in color_map_hex.items()}
                            default_rgba = (0, 0, 0, 0)
                            rgba_image = np.zeros((data_raster.shape[0], data_raster.shape[1], 4), dtype=np.uint8)

                            for code, rgba in color_map_rgba.items():
                                mask = (data_raster == code) & (~np.isnan(data_raster))
                                rgba_image[mask] = rgba
                            
                            if nodata_val is not None:
                                if np.isnan(nodata_val): nodata_mask = np.isnan(data_raster)
                                else: nodata_mask = (data_raster == nodata_val)
                                rgba_image[nodata_mask] = default_rgba
                            
                            print("DEBUG: (map_utils._add_overlay_layer) Converting colored array to PNG bytes...")
                            buffer = BytesIO()
                            plt.imsave(buffer, rgba_image, format='png') # plt from import
                            buffer.seek(0)
                            png_bytes = buffer.read()
                            buffer.close()
                            print(f"DEBUG: (map_utils._add_overlay_layer) Generated PNG ({len(png_bytes)} bytes).")

                            png_base64 = base64.b64encode(png_bytes).decode('utf-8')
                            image_url = f"data:image/png;base64,{png_base64}"
                            print(f"DEBUG: (map_utils._add_overlay_layer) Created Data URL (approx len: {len(image_url)}).")
                        else:
                            print("Warning (map_utils._add_overlay_layer): No common codes found between raster data and legend. Cannot colorize.")
                    else:
                        print("Warning (map_utils._add_overlay_layer): Legend data or map_code column missing. Cannot colorize raster.")

                    if image_url: # Check if image_url was successfully created
                        print("DEBUG: (map_utils._add_overlay_layer) Adding ImageOverlay to map...")
                        img_overlay = ipyleaflet.ImageOverlay(
                            url=image_url,
                            bounds=overlay_bounds,
                            name=f'Raster: {basename}'
                        )
                        map_widget.add_layer(img_overlay)
                        print(f"DEBUG: (map_utils._add_overlay_layer) Successfully ADDED ImageOverlay.")
                        created_layer = img_overlay
                    else:
                        print("Warning (map_utils._add_overlay_layer): No image URL generated, skipping ImageOverlay.")
                except MemoryError:
                    print("ERROR (map_utils._add_overlay_layer): Insufficient memory to process raster for display.")
                except Exception as img_overlay_err:
                    print(f"!!! Error processing Raster Overlay: {img_overlay_err}")
                    traceback.print_exc()
                finally:
                    if 'data_raster' in locals() and data_raster is not None: del data_raster
                    if 'rgba_image' in locals() and rgba_image is not None: del rgba_image
            else:
                print("Warning (map_utils._add_overlay_layer): Skipping Raster overlay - valid EPSG:4326 bounds were not provided.")

        elif map_file_str.lower().endswith(vector_extensions):
            # geopandas availability should be checked by the caller or here
            # For this example, assuming vector_gdf is a valid GeoDataFrame if provided
            if vector_gdf is not None and not vector_gdf.empty:
                print(f"DEBUG: (map_utils._add_overlay_layer) Processing Vector Overlay for {basename}...")
                try:
                    geom_col = vector_gdf.geometry.name
                    attr_cols = [c for c in vector_gdf.columns if c != geom_col]
                    default_style = {'color': 'black', 'weight': 1, 'fillColor': '#808080', 'fillOpacity': 0.6}
                    default_hover = {'fillColor': 'red', 'fillOpacity': 0.8}
                    style_callback_func = None
                    layer_name = f'Vector: {basename}'

                    if attr_cols:
                        map_code_col_vec = attr_cols[0] # Use local var to avoid conflict
                        print(f"DEBUG: (map_utils._add_overlay_layer) Using vector column '{map_code_col_vec}' for styling.")
                        if map_code_col_vec in vector_gdf.columns:
                            if legend_data is not None and 'map_code' in legend_data.columns:
                                valid_codes_vec = legend_data['map_code'].dropna().unique()
                                vector_codes_present = vector_gdf[map_code_col_vec].dropna().unique()
                                common_codes_vec = np.intersect1d(valid_codes_vec, vector_codes_present, assume_unique=True)

                                if len(common_codes_vec) > 0:
                                    print(f"DEBUG: (map_utils._add_overlay_layer) Colorizing vector based on {len(common_codes_vec)} common codes.")
                                    color_map_hex_vec = generate_class_color_map(common_codes_vec)
                                    color_map_lookup_vec = {code: hex_c for code, hex_c in color_map_hex_vec.items()}
                                    fallback_color_vec = '#808080'

                                    def vector_style_callback(feature, cmap=color_map_lookup_vec, code_col_cb=map_code_col_vec): # Renamed var in CB
                                        code_val = feature['properties'].get(code_col_cb) # Renamed var in CB
                                        color_fill = cmap.get(code_val, fallback_color_vec) # Renamed var in CB
                                        return {'fillColor': color_fill, 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
                                    style_callback_func = vector_style_callback # Assign to outer scope var
                                    layer_name = f'Vector: {basename} (Colored)'
                                else:
                                    print("Warning (map_utils._add_overlay_layer): No common codes found for vector styling. Using default style.")
                            else:
                                print("Warning (map_utils._add_overlay_layer): Legend data missing for vector coloring. Using default style.")
                        else:
                            print(f"Warning (map_utils._add_overlay_layer): Styling column '{map_code_col_vec}' not found in vector. Using default style.")
                    else:
                        print("Warning (map_utils._add_overlay_layer): Vector has no attribute columns for styling. Using default style.")
                    
                    print("DEBUG: (map_utils._add_overlay_layer) Converting GDF to GeoJSON dict...")
                    vector_gdf_clean = vector_gdf[vector_gdf.is_valid & ~vector_gdf.is_empty]
                    if vector_gdf_clean.empty:
                        print("Warning (map_utils._add_overlay_layer): No valid geometries remain after cleaning GDF.")
                        return None
                    geojson_data = json.loads(vector_gdf_clean.to_json())
                    print("DEBUG: (map_utils._add_overlay_layer) Creating GeoJSON layer...")
                    vector_layer = ipyleaflet.GeoJSON(
                        data=geojson_data,
                        style_callback=style_callback_func,
                        style=default_style,
                        hover_style=default_hover,
                        name=layer_name
                    )
                    map_widget.add_layer(vector_layer)
                    print(f"DEBUG: (map_utils._add_overlay_layer) Successfully ADDED GeoJSON layer '{layer_name}'.")
                    created_layer = vector_layer

                except Exception as geojson_err:
                    print(f"!!! Error creating/adding GeoJSON layer: {geojson_err}")
                    traceback.print_exc()
            else:
                print("Warning (map_utils._add_overlay_layer): Skipping Vector overlay - valid GeoDataFrame was not provided or was empty.")
        else:
            print(f"Warning (map_utils._add_overlay_layer): File type not recognized for overlay: {basename}")

    except Exception as overlay_error:
        print(f"!!! Error during overlay processing: {overlay_error}")
        traceback.print_exc()

    if created_layer:
        print(f"DEBUG: (map_utils._add_overlay_layer) Overlay layer object '{getattr(created_layer, 'name', 'Unnamed')}' CREATED.")
    else:
        print(f"DEBUG: (map_utils._add_overlay_layer) Overlay layer object NOT created.")
    return created_layer