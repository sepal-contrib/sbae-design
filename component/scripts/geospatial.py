"""SBAE Geospatial Processing Module.

Contains functions for file I/O, area calculation, and point generation.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point


def is_raster_file(file_path: str) -> bool:
    """Check if file is a supported raster format."""
    raster_extensions = [".tif", ".tiff", ".img", ".hdr"]
    return Path(file_path).suffix.lower() in raster_extensions


def is_vector_file(file_path: str) -> bool:
    """Check if file is a supported vector format."""
    vector_extensions = [".shp", ".geojson", ".gpkg", ".kml"]
    return Path(file_path).suffix.lower() in vector_extensions


def extract_raster_colormap(file_path: str) -> Dict[int, str]:
    """Extract color palette from raster file.

    Args:
        file_path: Path to raster file

    Returns:
        Dictionary mapping class codes to hex color strings.
        Returns empty dict if no colormap is found.
    """
    colors = {}

    try:
        with rasterio.open(file_path) as raster:
            # Try to get the colormap from the raster
            colormap = raster.colormap(1)

            if colormap:
                for class_code, rgba in colormap.items():
                    # Convert RGBA tuple to hex color
                    # rasterio returns values in 0-255 range
                    r, g, b, a = rgba
                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    colors[class_code] = hex_color

    except Exception:
        # If we can't extract colors, return empty dict
        pass

    return colors


def get_color_palette(file_path: str, class_codes: List[int]) -> Dict[int, str]:
    """Get color palette for given class codes, extracting from file if possible.

    Args:
        file_path: Path to raster or vector file
        class_codes: List of class codes to get colors for

    Returns:
        Dictionary mapping class codes to hex color strings
    """
    # Default color palette (ECharts default colors)
    default_colors = [
        "#5470c6",
        "#91cc75",
        "#fac858",
        "#ee6666",
        "#73c0de",
        "#3ba272",
        "#fc8452",
        "#9a60b4",
        "#ea7ccc",
    ]

    # Try to extract colors from raster file
    extracted_colors = {}
    if is_raster_file(file_path):
        extracted_colors = extract_raster_colormap(file_path)

    # Build color mapping for each class code
    color_map = {}
    for idx, class_code in enumerate(sorted(class_codes)):
        if class_code in extracted_colors:
            # Use extracted color if available
            color_map[class_code] = extracted_colors[class_code]
        else:
            # Fall back to default palette
            color_map[class_code] = default_colors[idx % len(default_colors)]

    return color_map


def compute_area_from_raster(file_path: str) -> pd.DataFrame:
    """Compute area for each class in a raster file.

    Args:
        file_path: Path to raster file

    Returns:
        DataFrame with columns: map_code, map_area, map_edited_class

    Raises:
        ValueError: If file cannot be read or processed
    """
    try:
        with rasterio.open(file_path) as raster:
            data = raster.read(1)
            transform = raster.transform

            # Calculate pixel area
            pixel_area = abs(transform.a * transform.e)

            # Count pixels per class (excluding nodata)
            nodata_value = raster.nodata if raster.nodata is not None else -9999
            valid_data = data[data != nodata_value]

            if len(valid_data) == 0:
                raise ValueError("No valid data found in raster")

            unique_values, counts = np.unique(valid_data, return_counts=True)

            # Calculate areas
            areas = counts * pixel_area

            return pd.DataFrame(
                {
                    "map_code": unique_values,
                    "map_area": areas,
                    "map_edited_class": [
                        f"Class {int(code)}" for code in unique_values
                    ],
                }
            )

    except Exception as e:
        raise ValueError(f"Error processing raster file: {str(e)}")


def compute_area_from_vector(file_path: str) -> pd.DataFrame:
    """Compute area for each class in a vector file.

    Args:
        file_path: Path to vector file

    Returns:
        DataFrame with columns: map_code, map_area, map_edited_class

    Raises:
        ValueError: If file cannot be read or no suitable class column found
    """
    try:
        gdf = gpd.read_file(file_path)

        if len(gdf) == 0:
            raise ValueError("Vector file contains no features")

        # Find the first non-geometry column as class column
        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                # Check if column has reasonable values for classification
                unique_vals = gdf[col].dropna().unique()
                if (
                    len(unique_vals) > 0 and len(unique_vals) <= 50
                ):  # Reasonable number of classes
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found in vector file")

        # Calculate areas (ensure we're in a projected CRS for accurate areas)
        if gdf.crs and gdf.crs.is_geographic:
            # Convert to appropriate UTM zone for area calculation
            gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
            areas = gdf_projected.geometry.area
        else:
            areas = gdf.geometry.area

        # Group by class and sum areas
        gdf_with_areas = gdf.copy()
        gdf_with_areas["area"] = areas
        area_by_class = gdf_with_areas.groupby(class_column)["area"].sum()

        return pd.DataFrame(
            {
                "map_code": area_by_class.index,
                "map_area": area_by_class.values,
                "map_edited_class": [f"Class {code}" for code in area_by_class.index],
            }
        )

    except Exception as e:
        raise ValueError(f"Error processing vector file: {str(e)}")


def compute_file_areas(file_path: str) -> pd.DataFrame:
    """Automatically detect file type and compute areas.

    Args:
        file_path: Path to classification file

    Returns:
        DataFrame with area information

    Raises:
        ValueError: If file format is not supported
    """
    if is_raster_file(file_path):
        return compute_area_from_raster(file_path)
    elif is_vector_file(file_path):
        return compute_area_from_vector(file_path)
    else:
        file_ext = Path(file_path).suffix.lower()
        raise ValueError(f"Unsupported file format: {file_ext}")


def save_uploaded_file(file_info, temp_dir: Optional[str] = None) -> str:
    """Save uploaded file to temporary directory.

    Args:
        file_info: FileInfo object from Solara FileDrop
        temp_dir: Optional temporary directory (created if None)

    Returns:
        Path to saved file
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    file_path = os.path.join(temp_dir, file_info["name"])

    with open(file_path, "wb") as f:
        file_info["file_obj"].seek(0)
        f.write(file_info["file_obj"].read())

    return file_path


def generate_sample_points_raster(
    file_path: str,
    samples_per_class: Dict[int, int],
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate stratified random sample points from raster data.

    Args:
        file_path: Path to raster file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility (None for random)

    Returns:
        DataFrame with sample points (longitude, latitude, map_code, map_edited_class)
        Points are always in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        with rasterio.open(file_path) as raster:
            data = raster.read(1)
            transform = raster.transform
            crs = raster.crs

            for class_code, n_samples in samples_per_class.items():
                if n_samples <= 0:
                    continue

                rows, cols = np.where(data == class_code)

                if len(rows) == 0:
                    print(f"Warning: No pixels found for class {class_code}")
                    continue

                n_available = len(rows)
                n_to_sample = min(n_samples, n_available)

                if n_to_sample > 0:
                    sampled_indices = np.random.choice(
                        n_available, n_to_sample, replace=False
                    )
                    sampled_rows = rows[sampled_indices]
                    sampled_cols = cols[sampled_indices]

                    for row, col in zip(sampled_rows, sampled_cols):
                        x, y = xy(transform, row + 0.5, col + 0.5)

                        if crs and not crs.is_geographic:
                            point_gdf = gpd.GeoDataFrame(
                                geometry=[Point(x, y)], crs=crs
                            )
                            point_gdf = point_gdf.to_crs("EPSG:4326")
                            lon, lat = (
                                point_gdf.geometry.iloc[0].x,
                                point_gdf.geometry.iloc[0].y,
                            )
                        else:
                            lon, lat = x, y

                        sample_points.append(
                            {
                                "longitude": lon,
                                "latitude": lat,
                                "map_code": class_code,
                                "map_edited_class": class_lookup.get(
                                    class_code, f"Class {class_code}"
                                ),
                            }
                        )

    except Exception as e:
        raise ValueError(f"Error generating points from raster: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_sample_points_vector(
    file_path: str,
    samples_per_class: Dict[int, int],
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate stratified random sample points from vector data.

    Args:
        file_path: Path to vector file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility (None for random)

    Returns:
        DataFrame with sample points (longitude, latitude, map_code, map_edited_class)
        Points are always in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        gdf = gpd.read_file(file_path)

        # Find the class column (same logic as in area computation)
        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                unique_vals = gdf[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found")

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = gdf
        elif not gdf.crs.is_geographic:
            gdf_geo = gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = gdf.copy()

        for class_code, n_samples in samples_per_class.items():
            if n_samples <= 0:
                continue

            # Filter geometries for this class
            class_geometries = gdf_geo[gdf_geo[class_column] == class_code]

            if class_geometries.empty:
                print(f"Warning: No geometries found for class {class_code}")
                continue

            # Generate random points within geometries
            samples_generated = 0
            max_attempts = n_samples * 100

            # Get bounds for efficiency
            bounds = class_geometries.total_bounds
            minx, miny, maxx, maxy = bounds

            for attempt in range(max_attempts):
                if samples_generated >= n_samples:
                    break

                # Generate random point within bounds
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                point = Point(x, y)

                # Check if point falls within any class geometry
                if any(geom.contains(point) for geom in class_geometries.geometry):
                    sample_points.append(
                        {
                            "longitude": x,
                            "latitude": y,
                            "map_code": class_code,
                            "map_edited_class": class_lookup.get(
                                class_code, f"Class {class_code}"
                            ),
                        }
                    )
                    samples_generated += 1

            if samples_generated < n_samples:
                print(
                    f"Warning: Only generated {samples_generated}/{n_samples} points for class {class_code}"
                )

    except Exception as e:
        raise ValueError(f"Error generating points from vector: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_simple_random_points_raster(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate simple random sample points from raster (no stratification).

    Args:
        file_path: Path to raster file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        with rasterio.open(file_path) as raster:
            data = raster.read(1)
            transform = raster.transform
            crs = raster.crs

            valid_mask = (
                data != raster.nodata
                if raster.nodata is not None
                else np.ones_like(data, dtype=bool)
            )
            rows, cols = np.where(valid_mask)

            if len(rows) == 0:
                raise ValueError("No valid pixels found in raster")

            n_available = len(rows)
            n_to_sample = min(total_samples, n_available)

            sampled_indices = np.random.choice(n_available, n_to_sample, replace=False)
            sampled_rows = rows[sampled_indices]
            sampled_cols = cols[sampled_indices]

            for row, col in zip(sampled_rows, sampled_cols):
                x, y = xy(transform, row + 0.5, col + 0.5)
                class_code = int(data[row, col])

                if crs and not crs.is_geographic:
                    point_gdf = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=crs)
                    point_gdf = point_gdf.to_crs("EPSG:4326")
                    lon, lat = (
                        point_gdf.geometry.iloc[0].x,
                        point_gdf.geometry.iloc[0].y,
                    )
                else:
                    lon, lat = x, y

                sample_points.append(
                    {
                        "longitude": lon,
                        "latitude": lat,
                        "map_code": class_code,
                        "map_edited_class": class_lookup.get(
                            class_code, f"Class {class_code}"
                        ),
                    }
                )

    except Exception as e:
        raise ValueError(f"Error generating simple random points from raster: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_systematic_points_raster(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate systematic sample points from raster (grid-based sampling).

    Args:
        file_path: Path to raster file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for starting point offset

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        with rasterio.open(file_path) as raster:
            data = raster.read(1)
            transform = raster.transform
            crs = raster.crs
            height, width = data.shape

            total_pixels = height * width
            grid_interval = int(np.sqrt(total_pixels / total_samples))
            if grid_interval < 1:
                grid_interval = 1

            offset_row = (
                np.random.randint(0, grid_interval)
                if seed is not None
                else grid_interval // 2
            )
            offset_col = (
                np.random.randint(0, grid_interval)
                if seed is not None
                else grid_interval // 2
            )

            for row in range(offset_row, height, grid_interval):
                for col in range(offset_col, width, grid_interval):
                    if len(sample_points) >= total_samples:
                        break

                    if raster.nodata is not None and data[row, col] == raster.nodata:
                        continue

                    x, y = xy(transform, row + 0.5, col + 0.5)
                    class_code = int(data[row, col])

                    if crs and not crs.is_geographic:
                        point_gdf = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=crs)
                        point_gdf = point_gdf.to_crs("EPSG:4326")
                        lon, lat = (
                            point_gdf.geometry.iloc[0].x,
                            point_gdf.geometry.iloc[0].y,
                        )
                    else:
                        lon, lat = x, y

                    sample_points.append(
                        {
                            "longitude": lon,
                            "latitude": lat,
                            "map_code": class_code,
                            "map_edited_class": class_lookup.get(
                                class_code, f"Class {class_code}"
                            ),
                        }
                    )

                if len(sample_points) >= total_samples:
                    break

    except Exception as e:
        raise ValueError(f"Error generating systematic points from raster: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_simple_random_points_vector(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate simple random sample points from vector (no stratification).

    Args:
        file_path: Path to vector file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        gdf = gpd.read_file(file_path)

        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                unique_vals = gdf[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found")

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = gdf
        elif not gdf.crs.is_geographic:
            gdf_geo = gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = gdf.copy()

        bounds = gdf_geo.total_bounds
        minx, miny, maxx, maxy = bounds

        samples_generated = 0
        max_attempts = total_samples * 100

        for attempt in range(max_attempts):
            if samples_generated >= total_samples:
                break

            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)

            for idx, geom in enumerate(gdf_geo.geometry):
                if geom.contains(point):
                    class_code = int(gdf_geo.iloc[idx][class_column])
                    sample_points.append(
                        {
                            "longitude": x,
                            "latitude": y,
                            "map_code": class_code,
                            "map_edited_class": class_lookup.get(
                                class_code, f"Class {class_code}"
                            ),
                        }
                    )
                    samples_generated += 1
                    break

    except Exception as e:
        raise ValueError(f"Error generating simple random points from vector: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_systematic_points_vector(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate systematic sample points from vector (grid-based sampling).

    Args:
        file_path: Path to vector file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for starting point offset

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        gdf = gpd.read_file(file_path)

        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                unique_vals = gdf[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found")

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = gdf
        elif not gdf.crs.is_geographic:
            gdf_geo = gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = gdf.copy()

        bounds = gdf_geo.total_bounds
        minx, miny, maxx, maxy = bounds

        area = (maxx - minx) * (maxy - miny)
        grid_spacing = np.sqrt(area / total_samples)

        offset_x = (
            np.random.uniform(0, grid_spacing) if seed is not None else grid_spacing / 2
        )
        offset_y = (
            np.random.uniform(0, grid_spacing) if seed is not None else grid_spacing / 2
        )

        x_coords = np.arange(minx + offset_x, maxx, grid_spacing)
        y_coords = np.arange(miny + offset_y, maxy, grid_spacing)

        for x in x_coords:
            for y in y_coords:
                if len(sample_points) >= total_samples:
                    break

                point = Point(x, y)

                for idx, geom in enumerate(gdf_geo.geometry):
                    if geom.contains(point):
                        class_code = int(gdf_geo.iloc[idx][class_column])
                        sample_points.append(
                            {
                                "longitude": x,
                                "latitude": y,
                                "map_code": class_code,
                                "map_edited_class": class_lookup.get(
                                    class_code, f"Class {class_code}"
                                ),
                            }
                        )
                        break

            if len(sample_points) >= total_samples:
                break

    except Exception as e:
        raise ValueError(f"Error generating systematic points from vector: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_sample_points(
    file_path: str,
    samples_per_class: Dict[int, int],
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
    sampling_method: str = "stratified",
    total_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Automatically detect file type and generate sample points.

    Args:
        file_path: Path to classification file
        samples_per_class: Dictionary of samples needed per class (empty for simple/systematic)
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility (None for random)
        sampling_method: "stratified", "simple", or "systematic"
        total_samples: Total samples to generate (for simple/systematic methods)

    Returns:
        DataFrame with sample points

    Raises:
        ValueError: If file format is not supported
    """
    # For simple random or systematic sampling, use non-stratified methods
    if sampling_method == "simple" and total_samples:
        if is_raster_file(file_path):
            return generate_simple_random_points_raster(
                file_path, total_samples, class_lookup, seed
            )
        elif is_vector_file(file_path):
            return generate_simple_random_points_vector(
                file_path, total_samples, class_lookup, seed
            )
    elif sampling_method == "systematic" and total_samples:
        if is_raster_file(file_path):
            return generate_systematic_points_raster(
                file_path, total_samples, class_lookup, seed
            )
        elif is_vector_file(file_path):
            return generate_systematic_points_vector(
                file_path, total_samples, class_lookup, seed
            )

    # Default to stratified sampling
    if is_raster_file(file_path):
        return generate_sample_points_raster(
            file_path, samples_per_class, class_lookup, seed
        )
    elif is_vector_file(file_path):
        return generate_sample_points_vector(
            file_path, samples_per_class, class_lookup, seed
        )
    else:
        file_ext = Path(file_path).suffix.lower()
        raise ValueError(f"Unsupported file format: {file_ext}")


def export_points_to_csv(points_df: pd.DataFrame) -> str:
    """Export points to CSV format.

    Args:
        points_df: DataFrame with sample points

    Returns:
        CSV string
    """
    return points_df.to_csv(index=False)


def export_points_to_geojson(points_df: pd.DataFrame) -> str:
    """Export points to GeoJSON format.

    Args:
        points_df: DataFrame with sample points

    Returns:
        GeoJSON string
    """
    gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
        crs="EPSG:4326",
    )
    return gdf.to_json()


def get_file_info(file_path: str) -> Dict:
    """Get basic information about a geospatial file.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information
    """
    info = {
        "file_type": "unknown",
        "size_mb": Path(file_path).stat().st_size / (1024 * 1024),
        "crs": None,
        "bounds": None,
        "feature_count": 0,
    }

    def get_crs_string(crs):
        """Extract clean CRS representation."""
        if not crs:
            return None
        epsg = crs.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        # Try to extract EPSG from the CRS string representation
        crs_str = str(crs)
        if 'AUTHORITY["EPSG"' in crs_str:
            # Extract EPSG code from WKT string like AUTHORITY["EPSG","3116"]
            import re

            match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', crs_str)
            if match:
                return f"EPSG:{match.group(1)}"
        return "Custom CRS"

    try:
        if is_raster_file(file_path):
            with rasterio.open(file_path) as raster:
                info["file_type"] = "raster"
                info["crs"] = get_crs_string(raster.crs)
                info["bounds"] = list(raster.bounds)
                info["width"] = raster.width
                info["height"] = raster.height
                info["feature_count"] = raster.width * raster.height

        elif is_vector_file(file_path):
            gdf = gpd.read_file(file_path)
            info["file_type"] = "vector"
            info["crs"] = get_crs_string(gdf.crs)
            info["bounds"] = list(gdf.total_bounds)
            info["feature_count"] = len(gdf)

    except Exception as e:
        info["error"] = str(e)

    return info
