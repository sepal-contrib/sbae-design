"""Tests for compute_area_from_raster (block-windowed class area counting)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from component.scripts.geospatial import compute_area_from_raster


def _write_raster(path, data, *, nodata=None, crs="EPSG:32633"):
    height, width = data.shape
    transform = from_bounds(500000, 4500000, 510000, 4510000, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        # small blocks so the windowed/accumulation path is exercised
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(data, 1)
    # pixel area in CRS units (10000m / width) * (10000m / height)
    return abs(transform.a * transform.e)


def test_counts_match_full_array():
    """Block-windowed counts match a naive np.unique over the whole band."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 6, size=(50, 70), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "classes.tif"
        pixel_area = _write_raster(path, data)

        df = compute_area_from_raster(str(path)).sort_values("map_code")

    values, counts = np.unique(data, return_counts=True)
    assert df["map_code"].tolist() == values.tolist()
    np.testing.assert_allclose(df["map_area"].to_numpy(), counts * pixel_area)


def test_nodata_excluded():
    """Declared nodata is dropped; other classes keep full counts."""
    data = np.full((20, 20), 3, dtype=np.uint8)
    data[:5, :] = 255  # 100 nodata pixels

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nodata.tif"
        pixel_area = _write_raster(path, data, nodata=255)

        df = compute_area_from_raster(str(path))

    assert df["map_code"].tolist() == [3]
    np.testing.assert_allclose(df["map_area"].iloc[0], 300 * pixel_area)


def test_float_class_map_uses_unique_fallback():
    """Float-valued class maps go through the per-block unique path."""
    data = np.array([[1.5, 1.5], [2.5, 2.5]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "float.tif"
        pixel_area = _write_raster(path, data)

        df = compute_area_from_raster(str(path)).sort_values("map_code")

    assert df["map_code"].tolist() == [1.5, 2.5]
    np.testing.assert_allclose(df["map_area"].to_numpy(), [2 * pixel_area] * 2)


def test_all_nodata_raises():
    data = np.full((10, 10), 0, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.tif"
        _write_raster(path, data, nodata=0)

        with pytest.raises(ValueError, match="No valid data"):
            compute_area_from_raster(str(path))
