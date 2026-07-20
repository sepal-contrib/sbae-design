import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from component.scripts.geospatial import extract_map_codes


def _write(path, data, crs, transform, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_samples_expected_classes(tmp_path):
    data = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    p = tmp_path / "clas.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 4, 1, 1))
    df = pd.DataFrame(
        {"x": [0.5, 2.5], "y": [3.5, 0.5]}
    )  # -> data[0][0]=1, data[3][2]=4
    out, dropped = extract_map_codes(df, str(p), "x", "y")
    assert dropped == 0
    assert out["map_code"].tolist() == [1, 4]


def test_drops_out_of_bounds(tmp_path):
    data = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    p = tmp_path / "clas.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 2, 1, 1))
    df = pd.DataFrame({"x": [0.5, 100.0], "y": [1.5, 100.0]})
    out, dropped = extract_map_codes(df, str(p), "x", "y")
    assert dropped == 1
    assert out["map_code"].tolist() == [1]


def test_keeps_all_points_when_drop_missing_false(tmp_path):
    # Design path: the sample is fixed, so keep every point -- sampled ones get
    # their real class, unsampleable ones fall back to their prior map_code.
    data = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    p = tmp_path / "clas.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 2, 1, 1))
    df = pd.DataFrame({"x": [0.5, 100.0], "y": [1.5, 100.0], "map_code": [0, 0]})
    out, dropped = extract_map_codes(df, str(p), "x", "y", drop_missing=False)
    assert dropped == 1  # one point was unsampleable
    assert len(out) == 2  # but nothing dropped
    assert out["map_code"].tolist() == [1, 0]  # sampled -> 1, missing -> prior 0


def test_drop_missing_false_without_prior_map_code_falls_back_to_zero(tmp_path):
    data = np.array([[7, 7], [7, 7]], dtype=np.uint8)
    p = tmp_path / "clas.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 2, 1, 1))
    df = pd.DataFrame({"x": [0.5, 100.0], "y": [1.5, 100.0]})  # no map_code column
    out, dropped = extract_map_codes(df, str(p), "x", "y", drop_missing=False)
    assert dropped == 1
    assert out["map_code"].tolist() == [7, 0]  # missing -> 0 when no prior column


def test_drops_nodata(tmp_path):
    data = np.array([[5, 255], [5, 5]], dtype=np.uint8)
    p = tmp_path / "clas.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 2, 1, 1), nodata=255)
    df = pd.DataFrame({"x": [1.5], "y": [1.5]})  # -> data[0][1]=255 (nodata)
    out, dropped = extract_map_codes(df, str(p), "x", "y")
    assert dropped == 1
    assert out.empty


def test_drops_point_on_right_bottom_edge(tmp_path):
    data = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    p = tmp_path / "edge.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 2, 1, 1))  # bounds x[0,2] y[0,2]
    # (2.0, 1.0): x == right edge -> maps to col 2 (out of range) -> must be dropped
    df = pd.DataFrame({"x": [2.0, 0.5], "y": [1.0, 0.5]})
    out, dropped = extract_map_codes(df, str(p), "x", "y")
    assert dropped == 1
    assert out["map_code"].tolist() == [1]


def test_existing_map_code_is_overwritten(tmp_path):
    data = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    p = tmp_path / "clas.tif"
    _write(p, data, "EPSG:4326", from_origin(0, 2, 1, 1))
    df = pd.DataFrame({"x": [0.5], "y": [1.5], "map_code": [99]})  # bogus pre-existing
    out, dropped = extract_map_codes(df, str(p), "x", "y")
    assert dropped == 0
    assert out["map_code"].tolist() == [1]  # raster value wins, 99 overwritten


def test_reprojects_points(tmp_path):
    # Raster in Web Mercator, placed ~500km from the coordinate origin so a
    # broken implementation that forgot to reproject (used raw lon/lat as if
    # already in EPSG:3857 metres) could never land inside the footprint by
    # coincidence -- only a correct reprojection does. The point below is the
    # mid-pixel of cell (row0, col0) -- x,y = (500500, 500500) in EPSG:3857 --
    # converted to EPSG:4326 via the spherical Web Mercator inverse formula
    # and cross-checked against rasterio.warp.transform's forward transform
    # (both agree to sub-metre precision; see task-1-report.md for the
    # derivation). Using the raw (4.496068, 4.491461) directly as metres
    # would fall far outside the raster's [500000, 502000] x [499000, 501000]
    # bounds, so a non-reprojecting implementation fails this test (dropped
    # == 1) rather than passing it by accident.
    data = np.array([[7, 7], [7, 7]], dtype=np.uint8)
    p = tmp_path / "merc.tif"
    _write(p, data, "EPSG:3857", from_origin(500000, 501000, 1000, 1000))
    df = pd.DataFrame({"x": [4.496068], "y": [4.491461]})
    out, dropped = extract_map_codes(df, str(p), "x", "y")
    assert dropped == 0
    assert out["map_code"].tolist() == [7]
