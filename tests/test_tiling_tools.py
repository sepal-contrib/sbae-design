"""GDAL CLI lookup must reach the venv's bin/ even when it is off PATH.

SEPAL launches the app from a micromamba venv by absolute path, so the binaries
sitting next to ``sys.executable`` are invisible to a bare-name lookup and the
COG route would be skipped for the slower rasterio fallback.
"""

import sys

from component.scripts.tiling import _find_tool, _gdal_ok, _tool


def test_find_tool_prefers_path(monkeypatch, tmp_path):
    on_path = tmp_path / "usr" / "bin"
    on_path.mkdir(parents=True)
    (on_path / "gdalinfo").touch(mode=0o755)
    monkeypatch.setenv("PATH", str(on_path))

    assert _find_tool("gdalinfo") == str(on_path / "gdalinfo")


def test_find_tool_falls_back_to_interpreter_sibling(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "gdal_translate").touch(mode=0o755)
    monkeypatch.setattr(sys, "executable", str(bindir / "python3"))
    monkeypatch.setenv("PATH", "/nonexistent")

    assert _find_tool("gdal_translate") == str(bindir / "gdal_translate")


def test_find_tool_returns_none_when_missing(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()  # no sibling binary
    monkeypatch.setattr(sys, "executable", str(bindir / "python3"))
    monkeypatch.setenv("PATH", "/nonexistent")

    assert _find_tool("gdaladdo") is None


def test_tool_falls_back_to_the_bare_name(monkeypatch, tmp_path):
    # subprocess then raises FileNotFoundError naming the tool, as before.
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python3"))
    monkeypatch.setenv("PATH", "/nonexistent")

    assert _tool("gdalwarp") == "gdalwarp"


def test_gdal_ok_is_false_when_a_tool_is_missing(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "gdalinfo").touch(mode=0o755)  # only one of the three
    monkeypatch.setattr(sys, "executable", str(bindir / "python3"))
    monkeypatch.setenv("PATH", "/nonexistent")

    assert _gdal_ok() is False


def test_gdal_ok_is_true_from_the_interpreter_sibling(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("gdalinfo", "gdal_translate", "gdaladdo"):
        (bindir / name).touch(mode=0o755)
    monkeypatch.setattr(sys, "executable", str(bindir / "python3"))
    monkeypatch.setenv("PATH", "/nonexistent")

    # the SEPAL case: nothing on PATH, everything next to the interpreter
    assert _gdal_ok() is True
