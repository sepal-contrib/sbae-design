"""Test coordinate transformation in point generation functions."""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from component.scripts.geospatial import (
    generate_sample_points,
    generate_simple_random_points_raster,
    generate_systematic_points_raster,
)


def test_raster_coordinate_transformation():
    """Test that raster point generation transforms to EPSG:4326."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = Path(tmpdir) / "test_raster.tif"

        width, height = 100, 100
        projected_crs = "EPSG:32633"

        transform = from_bounds(500000, 4500000, 510000, 4510000, width, height)

        data = np.random.randint(1, 4, (height, width), dtype=np.uint8)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=projected_crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        samples_per_class = {1: 10, 2: 10, 3: 10}
        class_lookup = {1: "Class 1", 2: "Class 2", 3: "Class 3"}

        points_df = generate_sample_points(
            str(raster_path), samples_per_class, class_lookup, seed=42
        )

        assert not points_df.empty, "No points generated"
        assert "longitude" in points_df.columns
        assert "latitude" in points_df.columns

        assert (
            points_df["longitude"].between(-180, 180).all()
        ), "Longitude values outside valid range"
        assert (
            points_df["latitude"].between(-90, 90).all()
        ), "Latitude values outside valid range"

        gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
            crs="EPSG:4326",
        )

        projected_gdf = gdf.to_crs(projected_crs)
        bounds = projected_gdf.total_bounds
        expected_bounds = [500000, 4500000, 510000, 4510000]

        assert bounds[0] >= expected_bounds[0] - 100, "Points outside expected bounds"
        assert bounds[1] >= expected_bounds[1] - 100, "Points outside expected bounds"
        assert bounds[2] <= expected_bounds[2] + 100, "Points outside expected bounds"
        assert bounds[3] <= expected_bounds[3] + 100, "Points outside expected bounds"

        print("✓ Raster coordinate transformation test passed")


def test_vector_coordinate_transformation():
    """Test that vector point generation transforms to EPSG:4326."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_path = Path(tmpdir) / "test_vector.gpkg"

        projected_crs = "EPSG:32633"

        geometries = [
            box(500000, 4500000, 503000, 4503000),
            box(503000, 4500000, 506000, 4503000),
            box(506000, 4500000, 510000, 4503000),
        ]

        gdf = gpd.GeoDataFrame(
            {"class_id": [1, 2, 3]}, geometry=geometries, crs=projected_crs
        )
        gdf.to_file(vector_path, driver="GPKG")

        samples_per_class = {1: 10, 2: 10, 3: 10}
        class_lookup = {1: "Class 1", 2: "Class 2", 3: "Class 3"}

        points_df = generate_sample_points(
            str(vector_path), samples_per_class, class_lookup, seed=42
        )

        assert not points_df.empty, "No points generated"
        assert "longitude" in points_df.columns
        assert "latitude" in points_df.columns

        assert (
            points_df["longitude"].between(-180, 180).all()
        ), "Longitude values outside valid range"
        assert (
            points_df["latitude"].between(-90, 90).all()
        ), "Latitude values outside valid range"

        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
            crs="EPSG:4326",
        )

        projected_points = points_gdf.to_crs(projected_crs)
        bounds = projected_points.total_bounds

        assert bounds[0] >= 500000 - 100, "Points outside expected bounds"
        assert bounds[1] >= 4500000 - 100, "Points outside expected bounds"
        assert bounds[2] <= 510000 + 100, "Points outside expected bounds"
        assert bounds[3] <= 4503000 + 100, "Points outside expected bounds"

        print("✓ Vector coordinate transformation test passed")


def test_simple_random_coordinate_transformation():
    """Test simple random sampling coordinate transformation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = Path(tmpdir) / "test_raster_simple.tif"

        width, height = 50, 50
        projected_crs = "EPSG:32633"

        transform = from_bounds(500000, 4500000, 505000, 4505000, width, height)

        data = np.random.randint(1, 3, (height, width), dtype=np.uint8)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=projected_crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        class_lookup = {1: "Class 1", 2: "Class 2"}

        points_df = generate_simple_random_points_raster(
            str(raster_path), 20, class_lookup, seed=42
        )

        assert not points_df.empty, "No points generated"
        assert (
            points_df["longitude"].between(-180, 180).all()
        ), "Longitude values outside valid range"
        assert (
            points_df["latitude"].between(-90, 90).all()
        ), "Latitude values outside valid range"

        print("✓ Simple random coordinate transformation test passed")


def test_systematic_coordinate_transformation():
    """Test systematic sampling coordinate transformation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = Path(tmpdir) / "test_raster_systematic.tif"

        width, height = 50, 50
        projected_crs = "EPSG:32633"

        transform = from_bounds(500000, 4500000, 505000, 4505000, width, height)

        data = np.random.randint(1, 3, (height, width), dtype=np.uint8)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=projected_crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        class_lookup = {1: "Class 1", 2: "Class 2"}

        points_df = generate_systematic_points_raster(
            str(raster_path), 25, class_lookup, seed=42
        )

        assert not points_df.empty, "No points generated"
        assert (
            points_df["longitude"].between(-180, 180).all()
        ), "Longitude values outside valid range"
        assert (
            points_df["latitude"].between(-90, 90).all()
        ), "Latitude values outside valid range"

        print("✓ Systematic coordinate transformation test passed")


if __name__ == "__main__":
    test_raster_coordinate_transformation()
    test_vector_coordinate_transformation()
    test_simple_random_coordinate_transformation()
    test_systematic_coordinate_transformation()
    print("\n✅ All coordinate transformation tests passed!")
