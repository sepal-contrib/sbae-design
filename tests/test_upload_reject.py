"""_reject_reason: only a raster can serve as the classification map.

The upload used to accept vectors -- areas computed, but the map layer was built
with ``add_raster``, which opens with rasterio and raised ``RasterioIOError``
straight out of a ``use_effect``.
"""

from component.tile.upload import _reject_reason


def test_a_raster_is_accepted():
    assert _reject_reason({"file_type": "raster"}) is None


def test_a_vector_is_rejected():
    reason = _reject_reason({"file_type": "vector"})

    assert reason is not None
    assert "raster" in reason.lower()


def test_an_unopenable_file_is_rejected():
    assert _reject_reason({"file_type": "unknown"}) is not None


def test_a_read_error_is_reported_verbatim():
    # the reader's own message says more than "unsupported format" would
    assert _reject_reason({"error": "boom", "file_type": "raster"}) == "boom"
