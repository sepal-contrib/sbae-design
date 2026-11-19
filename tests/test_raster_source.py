"""raster_source: what a selected file must be, and how band/nodata are resolved."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from component.scripts.geospatial import compute_area_from_raster, get_color_palette
from component.scripts.raster_source import (
    NotThematicError,
    format_nodata,
    needs_vrt,
    parse_nodata,
    selection_reject_code,
    write_vrt,
)


def test_not_thematic_error_is_a_value_error_with_a_code():
    err = NotThematicError("no_valid_data")
    assert isinstance(err, ValueError)
    assert err.code == "no_valid_data"
    assert "No valid data" in str(err)


def test_not_thematic_error_appends_detail():
    assert "more than 255" in str(NotThematicError("too_many_classes", "more than 255"))


def test_selection_accepts_a_raster_with_a_crs():
    assert selection_reject_code({"file_type": "raster", "crs": "EPSG:4326"}) is None


def test_selection_rejects_a_vector():
    assert selection_reject_code({"file_type": "vector", "crs": "EPSG:4326"}) == (
        "not_a_raster"
    )


def test_selection_rejects_an_unopenable_file():
    assert selection_reject_code({"file_type": "unknown"}) == "not_a_raster"


def test_selection_rejects_a_raster_without_crs():
    assert selection_reject_code({"file_type": "raster", "crs": None}) == "no_crs"


def test_selection_reports_reader_errors_first():
    info = {"error": "boom", "file_type": "raster", "crs": None}
    assert selection_reject_code(info) == "read_error"


@pytest.mark.parametrize(
    "text,value",
    [
        ("", None),
        ("  ", None),
        (None, None),
        ("0", 0.0),
        ("255", 255.0),
        ("-9999", -9999.0),
        ("1.5", 1.5),
    ],
)
def test_parse_nodata(text, value):
    assert parse_nodata(text) == value


@pytest.mark.parametrize("text", ["abc", "nan", "inf", "1,5"])
def test_parse_nodata_rejects_non_numbers(text):
    with pytest.raises(ValueError):
        parse_nodata(text)


@pytest.mark.parametrize(
    "value,text",
    [
        (None, ""),
        (0.0, "0"),
        (255.0, "255"),
        (-9999.0, "-9999"),
        (1.5, "1.5"),
        (float("nan"), ""),
    ],
)
def test_format_nodata(value, text):
    assert format_nodata(value) == text


def test_needs_vrt_is_false_for_band_1_and_the_declared_nodata():
    assert needs_vrt({"nodata": 0.0}, 1, 0.0) is False
    assert needs_vrt({"nodata": None}, 1, None) is False


def test_needs_vrt_for_another_band():
    assert needs_vrt({"nodata": 0.0}, 2, 0.0) is True


def test_needs_vrt_for_another_nodata():
    assert needs_vrt({"nodata": 0.0}, 1, 5.0) is True
    assert needs_vrt({"nodata": None}, 1, 0.0) is True
    assert needs_vrt({"nodata": 0.0}, 1, None) is True


def test_needs_vrt_treats_nan_nodata_as_undeclared():
    assert needs_vrt({"nodata": float("nan")}, 1, None) is False


BASE = np.array([[0, 1, 1], [2, 2, 3], [3, 3, 5]], dtype="uint8")


def _write(path, bands, *, nodata=0, colormap=None):
    data = np.stack(bands)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500000, 4650000, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(data)
        if colormap:
            dst.write_colormap(1, colormap)
    return path


def test_vrt_selects_the_band(tmp_path):
    src = _write(tmp_path / "m.tif", [BASE, BASE * 2, BASE])

    vrt = write_vrt(src, 2, 0.0, tmp_path)

    with rasterio.open(vrt) as ds:
        assert ds.count == 1
        assert ds.nodata == 0.0
        assert ds.crs.to_epsg() == 32633
    assert compute_area_from_raster(vrt)["map_code"].tolist() == [2, 4, 6, 10]


def test_vrt_declares_nodata_on_a_file_that_has_none(tmp_path):
    src = _write(tmp_path / "n.tif", [BASE], nodata=None)
    assert compute_area_from_raster(str(src))["map_code"].tolist() == [0, 1, 2, 3, 5]

    vrt = write_vrt(src, 1, 0.0, tmp_path)

    assert compute_area_from_raster(vrt)["map_code"].tolist() == [1, 2, 3, 5]


def test_vrt_can_move_nodata_onto_a_class(tmp_path):
    src = _write(tmp_path / "b.tif", [BASE])  # declares nodata 0

    vrt = write_vrt(src, 1, 5.0, tmp_path)

    assert compute_area_from_raster(vrt)["map_code"].tolist() == [0, 1, 2, 3]


def test_vrt_can_clear_a_declared_nodata(tmp_path):
    src = _write(tmp_path / "c.tif", [BASE])

    vrt = write_vrt(src, 1, None, tmp_path)

    with rasterio.open(vrt) as ds:
        assert ds.nodata is None
    assert compute_area_from_raster(vrt)["map_code"].tolist() == [0, 1, 2, 3, 5]


def test_vrt_carries_the_colour_table(tmp_path):
    cmap = {
        1: (0, 100, 0, 255),
        2: (255, 255, 0, 255),
        3: (0, 0, 255, 255),
        5: (255, 0, 0, 255),
    }
    src = _write(tmp_path / "p.tif", [BASE], colormap=cmap)

    vrt = write_vrt(src, 1, 0.0, tmp_path)

    assert get_color_palette(vrt, [1, 2, 3, 5]) == {
        1: "#006400",
        2: "#ffff00",
        3: "#0000ff",
        5: "#ff0000",
    }


def test_vrt_is_named_after_the_source_and_overwritten(tmp_path):
    src = _write(tmp_path / "s.tif", [BASE, BASE])
    dest = tmp_path / "scratch"  # does not exist yet

    first = write_vrt(src, 2, 0.0, dest)
    second = write_vrt(src, 2, 0.0, dest)

    assert first == second == str(dest / "s.b2.vrt")


def test_vrt_rejects_a_band_out_of_range(tmp_path):
    src = _write(tmp_path / "r.tif", [BASE])
    with pytest.raises(ValueError):
        write_vrt(src, 3, 0.0, tmp_path)
