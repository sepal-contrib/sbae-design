"""_upload_toast: the terminal upload/optimization toast decision.

Guards the duplicate "optimized" toast: the raster success must fire only when a
real optimization result exists, not on the prep thread's idle FINISHED state.
"""

import solara

from component.tile.upload import _upload_toast

_R = solara.ResultState


def test_no_file_no_toast():
    assert (
        _upload_toast(
            has_file=False, is_raster=True, state=_R.FINISHED, value=None, error=None
        )
        is None
    )


def test_non_raster_success():
    level, msg = _upload_toast(
        has_file=True, is_raster=False, state=_R.FINISHED, value=None, error=None
    )
    assert level == "success"
    assert msg == "File uploaded successfully."


def test_raster_error():
    level, msg = _upload_toast(
        has_file=True, is_raster=True, state=_R.ERROR, value=None, error="boom"
    )
    assert level == "error"
    assert "boom" in msg


def test_raster_optimized_only_with_real_result():
    level, msg = _upload_toast(
        has_file=True,
        is_raster=True,
        state=_R.FINISHED,
        value={"path": "/tmp/opt.tif"},
        error=None,
    )
    assert level == "success"
    assert "optimized" in msg


def test_raster_finished_without_result_is_silent():
    # The no-op / idle FINISHED (lambda: None) must NOT toast -- this is the
    # duplicate-toast regression.
    assert (
        _upload_toast(
            has_file=True, is_raster=True, state=_R.FINISHED, value=None, error=None
        )
        is None
    )


def test_raster_running_is_silent():
    assert (
        _upload_toast(
            has_file=True, is_raster=True, state=_R.RUNNING, value=None, error=None
        )
        is None
    )
