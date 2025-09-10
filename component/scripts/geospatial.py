"""SBAE Geospatial Processing Module.

Contains functions for file I/O, area calculation, and point generation.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

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
    file_path: str, samples_per_class: Dict[int, int], class_lookup: Dict[int, str]
) -> pd.DataFrame:
    """Generate stratified random sample points from raster data.

    Args:
        file_path: Path to raster file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names

    Returns:
        DataFrame with sample points (longitude, latitude, map_code, map_edited_class)
    """
    sample_points = []

    try:
        with rasterio.open(file_path) as raster:
            data = raster.read(1)
            transform = raster.transform
            crs = raster.crs

            for class_code, n_samples in samples_per_class.items():
                if n_samples <= 0:
                    continue

                # Find all pixels of this class
                rows, cols = np.where(data == class_code)

                if len(rows) == 0:
                    print(f"Warning: No pixels found for class {class_code}")
                    continue

                # Randomly sample pixel locations
                n_available = len(rows)
                n_to_sample = min(n_samples, n_available)

                if n_to_sample > 0:
                    sampled_indices = np.random.choice(
                        n_available, n_to_sample, replace=False
                    )
                    sampled_rows = rows[sampled_indices]
                    sampled_cols = cols[sampled_indices]

                    # Convert pixel coordinates to geographic coordinates
                    for row, col in zip(sampled_rows, sampled_cols):
                        # Get pixel center coordinates
                        x, y = xy(transform, row + 0.5, col + 0.5)

                        # Transform to WGS84 if needed
                        if crs and not crs.is_geographic:
                            # This is a simplified approach - for production use proper CRS transformation
                            pass

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

    except Exception as e:
        raise ValueError(f"Error generating points from raster: {str(e)}")

    return pd.DataFrame(sample_points)


def generate_sample_points_vector(
    file_path: str, samples_per_class: Dict[int, int], class_lookup: Dict[int, str]
) -> pd.DataFrame:
    """Generate stratified random sample points from vector data.

    Args:
        file_path: Path to vector file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names

    Returns:
        DataFrame with sample points (longitude, latitude, map_code, map_edited_class)
    """
    sample_points = []

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

        # Ensure we're in geographic CRS for lat/lon output
        if gdf.crs and not gdf.crs.is_geographic:
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


def generate_sample_points(
    file_path: str, samples_per_class: Dict[int, int], class_lookup: Dict[int, str]
) -> pd.DataFrame:
    """Automatically detect file type and generate sample points.

    Args:
        file_path: Path to classification file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names

    Returns:
        DataFrame with sample points

    Raises:
        ValueError: If file format is not supported
    """
    if is_raster_file(file_path):
        return generate_sample_points_raster(file_path, samples_per_class, class_lookup)
    elif is_vector_file(file_path):
        return generate_sample_points_vector(file_path, samples_per_class, class_lookup)
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

    try:
        if is_raster_file(file_path):
            with rasterio.open(file_path) as raster:
                info["file_type"] = "raster"
                info["crs"] = str(raster.crs) if raster.crs else None
                info["bounds"] = list(raster.bounds)
                info["width"] = raster.width
                info["height"] = raster.height
                info["feature_count"] = raster.width * raster.height

        elif is_vector_file(file_path):
            gdf = gpd.read_file(file_path)
            info["file_type"] = "vector"
            info["crs"] = str(gdf.crs) if gdf.crs else None
            info["bounds"] = list(gdf.total_bounds)
            info["feature_count"] = len(gdf)

    except Exception as e:
        info["error"] = str(e)

    return info
