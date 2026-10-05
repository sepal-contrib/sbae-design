"""reject_reason / area_error_message: acceptance codes become one toast each."""

from component.scripts.raster_source import NotThematicError
from component.tile.upload import area_error_message, reject_reason


def test_a_raster_with_a_crs_is_accepted():
    assert reject_reason({"file_type": "raster", "crs": "EPSG:4326"}) is None


def test_a_vector_is_rejected():
    reason = reject_reason({"file_type": "vector", "crs": "EPSG:4326"})

    assert reason is not None
    assert "raster" in reason.lower()


def test_an_unopenable_file_is_rejected():
    assert reject_reason({"file_type": "unknown"}) is not None


def test_a_raster_without_crs_is_rejected():
    reason = reject_reason({"file_type": "raster", "crs": None})

    assert reason is not None
    assert "coordinate reference" in reason.lower()


def test_a_read_error_is_reported_verbatim():
    # the reader's own message says more than "unsupported format" would
    assert reject_reason({"error": "boom", "file_type": "raster"}) == "boom"


def test_not_thematic_codes_map_to_their_message():
    assert "non-integer" in area_error_message(NotThematicError("non_integral"))
    assert "255" in area_error_message(NotThematicError("too_many_classes"))
    assert "nodata" in area_error_message(NotThematicError("no_valid_data"))
    assert "coordinate" in area_error_message(NotThematicError("no_crs"))


def test_other_errors_pass_through():
    assert area_error_message(RuntimeError("boom")) == "boom"
