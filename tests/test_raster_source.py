"""raster_source: what a selected file must be, and how band/nodata are resolved."""

import pytest

from component.scripts.raster_source import (
    NotThematicError,
    format_nodata,
    needs_vrt,
    parse_nodata,
    selection_reject_code,
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
