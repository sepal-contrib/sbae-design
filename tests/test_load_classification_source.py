"""load_classification_source: the one call that turns a picked file into a design source."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from component.scripts.geospatial import (
    extract_map_codes,
    generate_sample_points,
    get_file_info,
    load_classification_source,
)
from component.scripts.raster_source import NotThematicError

BASE = np.array([[0, 1, 1], [2, 2, 3], [3, 3, 5]], dtype="uint8")

# Three views of the same Congo map, so picking a band visibly changes the
# design: band 1 is the original classification, band 2 collapses it to
# forest (1) / non-forest (2), band 3 offsets every code by 100. Geographic
# CRS, so these also exercise the VRT against the geodesic area path, which
# the synthetic UTM rasters above never reach.
MULTIBAND_MAP = Path(__file__).parent / "data" / "multiband_congo.tif"
MULTIBAND_CLASSES = {
    1: [2, 4, 11, 12, 13, 31, 32, 33, 34],
    2: [1, 2],
    3: [102, 104, 111, 112, 113, 131, 132, 133, 134],
}


def _write(path, bands, *, nodata=0, crs="EPSG:32633"):
    data = np.stack(bands)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs=crs,
        transform=from_origin(500000, 4650000, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


def test_file_info_describes_the_raster(tmp_path):
    path = _write(tmp_path / "m.tif", [BASE, BASE * 2])

    info = get_file_info(path)

    assert info["file_type"] == "raster"
    assert info["driver"] == "GTiff"
    assert info["band_count"] == 2
    assert info["dtype"] == "uint8"
    assert info["nodata"] == 0.0
    assert info["pixels"] == 9
    assert info["crs"] == "EPSG:32633"
    assert "feature_count" not in info


def test_keeps_the_original_path_for_band_1_and_declared_nodata(tmp_path):
    path = _write(tmp_path / "a.tif", [BASE])

    out = load_classification_source(
        path, band=1, nodata=0.0, temp_dir=str(tmp_path / "s")
    )

    assert out["path"] == path
    assert out["source_path"] == path
    assert out["band"] == 1
    assert out["nodata"] == 0.0
    assert out["area_method"] == "planar"
    assert out["area_data"]["map_code"].tolist() == [1, 2, 3, 5]
    assert set(out["color_palette"]) == {1, 2, 3, 5}


def test_default_nodata_is_the_declared_one(tmp_path):
    path = _write(tmp_path / "d.tif", [BASE])

    out = load_classification_source(path, temp_dir=str(tmp_path / "s"))

    assert out["path"] == path
    assert out["nodata"] == 0.0


def test_writes_a_vrt_for_another_band(tmp_path):
    path = _write(tmp_path / "b.tif", [BASE, BASE * 2])

    out = load_classification_source(
        path, band=2, nodata=0.0, temp_dir=str(tmp_path / "s")
    )

    assert out["path"] == str(tmp_path / "s" / "b.b2.vrt")
    assert out["source_path"] == path
    assert out["band"] == 2
    assert out["area_data"]["map_code"].tolist() == [2, 4, 6, 10]


def test_writes_a_vrt_for_another_nodata(tmp_path):
    path = _write(tmp_path / "n.tif", [BASE], nodata=None)

    out = load_classification_source(path, nodata=0.0, temp_dir=str(tmp_path / "s"))

    assert out["path"].endswith("n.b1.vrt")
    assert out["area_data"]["map_code"].tolist() == [1, 2, 3, 5]


def test_a_raster_without_crs_raises_no_crs(tmp_path):
    path = _write(tmp_path / "c.tif", [BASE], crs=None)

    with pytest.raises(NotThematicError) as info:
        load_classification_source(path, temp_dir=str(tmp_path / "s"))
    assert info.value.code == "no_crs"


def test_a_non_raster_raises_value_error(tmp_path):
    path = tmp_path / "v.geojson"
    path.write_text('{"type": "FeatureCollection", "features": []}')

    with pytest.raises(ValueError):
        load_classification_source(str(path), temp_dir=str(tmp_path / "s"))


def test_file_info_reports_a_missing_file_as_an_error(tmp_path):
    info = get_file_info(str(tmp_path / "gone.tif"))

    assert info["file_type"] == "unknown"
    assert "error" in info
    assert info["size_mb"] == 0.0


def test_the_multiband_map_offers_three_bands():
    info = get_file_info(str(MULTIBAND_MAP))

    assert info["band_count"] == 3
    assert info["dtype"] == "uint8"
    assert info["nodata"] == 0.0
    assert info["crs"] == "EPSG:4326"


@pytest.mark.parametrize("band", sorted(MULTIBAND_CLASSES))
def test_each_band_of_the_multiband_map_yields_its_own_classes(band, tmp_path):
    out = load_classification_source(
        str(MULTIBAND_MAP), band=band, temp_dir=str(tmp_path)
    )

    assert out["band"] == band
    assert out["area_data"]["map_code"].tolist() == MULTIBAND_CLASSES[band]
    assert set(out["color_palette"]) == set(MULTIBAND_CLASSES[band])
    # a geographic map, so its areas are ellipsoidal whichever band is picked
    assert out["area_method"] == "geodesic"
    assert out["area_data"]["map_area"].sum() > 0


def test_points_generated_through_a_band_vrt_read_back_that_band(tmp_path):
    out = load_classification_source(str(MULTIBAND_MAP), band=2, temp_dir=str(tmp_path))
    assert out["path"] != str(MULTIBAND_MAP)  # band 2 went through a VRT

    points = generate_sample_points(
        file_path=out["path"],
        samples_per_class={code: 5 for code in MULTIBAND_CLASSES[2]},
        class_lookup={},
        seed=42,
    )
    back, dropped = extract_map_codes(
        points.assign(map_code=0), out["path"], "longitude", "latitude"
    )

    assert len(points) == 10
    assert dropped == 0
    assert back["map_code"].tolist() == points["map_code"].tolist()
    assert set(back["map_code"]) <= set(MULTIBAND_CLASSES[2])
