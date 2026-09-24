"""Tests for compute_area_from_raster (block-windowed class area counting)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Geod, Transformer
from rasterio.transform import from_bounds, from_origin

from component.scripts.geospatial import (
    choose_area_method_for,
    compute_area_from_raster,
)
from component.scripts.raster_source import NotThematicError


def _write_raster(path, data, *, nodata=None, crs="EPSG:32633", transform=None):
    height, width = data.shape
    if transform is None:
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
    # pixel area in CRS units
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


def test_all_nodata_raises():
    data = np.full((10, 10), 0, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.tif"
        _write_raster(path, data, nodata=0)

        with pytest.raises(ValueError, match="No valid data"):
            compute_area_from_raster(str(path))


def test_float_class_map_with_integral_values_yields_int_codes(tmp_path):
    data = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    path = tmp_path / "float_int.tif"
    pixel_area = _write_raster(path, data)

    df = compute_area_from_raster(str(path)).sort_values("map_code")

    assert df["map_code"].tolist() == [1, 2]
    assert all(isinstance(code, (int, np.integer)) for code in df["map_code"])
    np.testing.assert_allclose(df["map_area"].to_numpy(), [2 * pixel_area] * 2)


def test_non_integral_values_are_rejected(tmp_path):
    data = np.array([[1.5, 1.5], [2.5, 2.5]], dtype=np.float32)
    path = tmp_path / "float.tif"
    _write_raster(path, data)

    with pytest.raises(NotThematicError) as info:
        compute_area_from_raster(str(path))
    assert info.value.code == "non_integral"


def test_nan_is_never_a_class(tmp_path):
    data = np.array([[1.0, np.nan], [2.0, 2.0]], dtype=np.float32)
    path = tmp_path / "nan.tif"
    _write_raster(path, data)

    df = compute_area_from_raster(str(path))

    assert df["map_code"].tolist() == [1, 2]


def test_a_raster_without_crs_is_rejected(tmp_path):
    data = np.ones((4, 4), dtype=np.uint8)
    path = tmp_path / "nocrs.tif"
    _write_raster(path, data, crs=None)

    with pytest.raises(NotThematicError) as info:
        compute_area_from_raster(str(path))
    assert info.value.code == "no_crs"


def test_too_many_classes_is_rejected(tmp_path):
    data = np.arange(1, 301, dtype=np.uint16).reshape(20, 15)
    path = tmp_path / "many.tif"
    _write_raster(path, data)

    with pytest.raises(NotThematicError) as info:
        compute_area_from_raster(str(path))
    assert info.value.code == "too_many_classes"


def test_wide_integer_codes_do_not_go_through_bincount(tmp_path):
    # bincount would allocate a 2e9-entry array for this code
    data = np.array([[1, 2], [2, 2_000_000_000]], dtype=np.int32)
    path = tmp_path / "wide.tif"
    pixel_area = _write_raster(path, data)

    df = compute_area_from_raster(str(path)).sort_values("map_code")

    assert df["map_code"].tolist() == [1, 2, 2_000_000_000]
    np.testing.assert_allclose(
        df["map_area"].to_numpy(), np.array([1, 2, 1]) * pixel_area
    )


def test_utm_areas_stay_planar_and_bit_identical(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(1, 4, size=(300, 300), dtype=np.uint8)
    path = tmp_path / "utm.tif"
    transform = from_origin(500_000, 4_650_000, 30, 30)
    _write_raster(path, data, transform=transform)

    df = compute_area_from_raster(str(path)).sort_values("map_code")

    counts = np.array([(data == code).sum() for code in (1, 2, 3)])
    assert df["map_area"].tolist() == (counts * 900.0).tolist()
    method, departure = choose_area_method_for(str(path))
    assert method == "planar"
    assert departure < 0.01


def test_survey_feet_areas_are_in_square_metres(tmp_path):
    data = np.ones((10, 10), dtype=np.uint8)
    path = tmp_path / "feet.tif"
    _write_raster(
        path,
        data,
        crs="EPSG:2227",
        transform=from_origin(6_000_000, 2_100_000, 100, 100),
    )

    df = compute_area_from_raster(str(path))

    assert df["map_area"].iloc[0] == pytest.approx(
        100 * 10_000 * 0.30480060960121924**2
    )


def _row_reference(path):
    """Exact per-class areas for a north-up geographic grid, one Geod call per row."""
    geod = Geod(ellps="WGS84")
    with rasterio.open(path) as src:
        t = src.transform
        arr = src.read(1)
        nodata = src.nodata
        row_area = np.empty(src.height)
        for r in range(src.height):
            top = t.f + r * t.e
            area, _ = geod.polygon_area_perimeter(
                [t.c, t.c + t.a, t.c + t.a, t.c], [top, top, top + t.e, top + t.e]
            )
            row_area[r] = abs(area)
    codes = [int(c) for c in np.unique(arr) if c != nodata]
    return {c: float(((arr == c).sum(axis=1) * row_area).sum()) for c in codes}


def test_geographic_areas_are_geodesic_within_a_tenth_of_a_percent(tmp_path):
    # classes in latitude bands: the adversarial layout for tile-mean weights
    data = np.zeros((300, 300), dtype=np.uint8)
    data[:100] = 1
    data[100:200] = 2
    data[200:] = 3
    data[:, :10] = 0  # a nodata stripe
    path = tmp_path / "geo.tif"
    _write_raster(
        path,
        data,
        nodata=0,
        crs="EPSG:4326",
        transform=from_origin(12.0, 45.0, 0.00027, 0.00027),
    )

    df = compute_area_from_raster(str(path))

    reference = _row_reference(str(path))
    got = dict(zip(df["map_code"], df["map_area"]))
    assert set(got) == {1, 2, 3}
    for code, area in reference.items():
        assert got[code] == pytest.approx(area, rel=1e-3), code
    assert choose_area_method_for(str(path))[0] == "geodesic"


def test_web_mercator_at_60n_is_geodesic(tmp_path):
    data = np.ones((300, 300), dtype=np.uint8)
    path = tmp_path / "merc.tif"
    transform = from_origin(1_335_833, 8_399_737, 60, 60)
    _write_raster(path, data, crs="EPSG:3857", transform=transform)

    df = compute_area_from_raster(str(path))

    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x0, x1 = transform.c, transform.c + 300 * transform.a
    y0, y1 = transform.f, transform.f + 300 * transform.e
    lon, lat = to_lonlat.transform([x0, x1, x1, x0], [y0, y0, y1, y1])
    footprint, _ = Geod(ellps="WGS84").polygon_area_perimeter(list(lon), list(lat))
    assert df["map_area"].iloc[0] == pytest.approx(abs(footprint), rel=5e-3)
    assert choose_area_method_for(str(path))[0] == "geodesic"
    # planar would have said 300 * 300 * 3600 m2, off by more than 3x here
    assert df["map_area"].iloc[0] < 0.5 * 300 * 300 * 3600
