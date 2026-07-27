import asyncio

import pandas as pd
import pytest

from component.scripts.vector_tiles import VectorTileError
from component.widget import map as map_mod
from component.widget.map import SbaeMap


class _FakeMap:
    def __init__(self):
        self.sample_points_layer = None
        self.reference_points_layer = None
        self.reference_points_dir = None
        self.added = []
        self.removed = []
        self.fitted = None

    def add_layer(self, layer, key=None):
        self.added.append((layer, key))

    def remove_layer(self, key, base=False, none_ok=False):
        self.removed.append(key)

    def fit_bounds(self, bounds):
        self.fitted = bounds


class _BoundedLayer:
    def __init__(self):
        self.bounds = [[0.0, 0.0], [1.0, 2.0]]


def test_attach_swaps_layer():
    m = _FakeMap()
    SbaeMap.attach_sample_points_layer(m, "L1")
    assert m.added == [("L1", "sample_pts")]
    assert m.sample_points_layer == "L1"
    SbaeMap.attach_sample_points_layer(m, "L2")
    assert m.removed == ["L1"]
    assert m.sample_points_layer == "L2"


def test_attach_zooms_to_layer_bounds():
    m = _FakeMap()
    SbaeMap.attach_sample_points_layer(m, _BoundedLayer())
    assert m.fitted == [[0.0, 0.0], [1.0, 2.0]]  # auto-zoomed to the points


def test_attach_skips_zoom_when_layer_has_no_bounds():
    m = _FakeMap()
    SbaeMap.attach_sample_points_layer(m, "L1")  # a bounds-less layer double
    assert m.fitted is None  # view left untouched


def test_build_empty_returns_none():
    layer = asyncio.run(
        SbaeMap.build_sample_points_layer(_FakeMap(), pd.DataFrame(), {})
    )
    assert layer is None


def test_build_delegates(monkeypatch):
    called = {}

    async def fake_build(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        called["colors"] = colors
        return "LAYER"

    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", fake_build)
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    layer = asyncio.run(
        SbaeMap.build_sample_points_layer(_FakeMap(), df, {3: "#333333"})
    )
    assert layer == "LAYER"
    assert called["colors"] == {3: "#333333"}


def test_add_sample_points_error_notifies(monkeypatch):
    from component.model import app_state as real_app_state

    async def boom(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        raise VectorTileError("nope")

    errs = []
    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", boom)
    monkeypatch.setattr(real_app_state, "add_error", lambda msg: errs.append(msg))
    m = _FakeMap()
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    asyncio.run(SbaeMap.add_sample_points(m, df))  # must not raise
    assert errs and "Could not render sample points" in errs[0]
    assert m.added == []


def test_attach_removes_previous_backing_dir(tmp_path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    m = _FakeMap()
    m.sample_points_layer = "OLD"
    m.sample_points_dir = str(old_dir)
    m._pending_points_dir = str(new_dir)

    SbaeMap.attach_sample_points_layer(m, "NEW")

    assert not old_dir.exists()  # previous layer's backing dir is reclaimed
    assert new_dir.exists()  # the dir backing the now-live layer survives
    assert m.sample_points_dir == str(new_dir)


def test_build_cleans_dir_on_failure(monkeypatch, tmp_path):
    built_dir = tmp_path / "built"

    def fake_mkdtemp(prefix=""):
        built_dir.mkdir()
        return str(built_dir)

    async def boom(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        raise VectorTileError("nope")

    monkeypatch.setattr(map_mod.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", boom)
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})

    with pytest.raises(VectorTileError):
        asyncio.run(SbaeMap.build_sample_points_layer(_FakeMap(), df, {}))

    assert not built_dir.exists()  # the just-created dir is not leaked


def test_build_sample_points_layer_passes_point_color(monkeypatch):
    captured = {}

    async def fake_build(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        captured["default_color"] = default_color
        return "LAYER"

    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", fake_build)
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    asyncio.run(
        SbaeMap.build_sample_points_layer(_FakeMap(), df, {}, point_color="#ff7f0e")
    )
    assert captured["default_color"] == "#ff7f0e"


def test_build_sample_points_layer_rides_overlay_pane(monkeypatch):
    class _Layer:
        pass

    async def fake_build(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        return _Layer()

    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", fake_build)
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    layer = asyncio.run(SbaeMap.build_sample_points_layer(_FakeMap(), df, {}))
    # points go in leaflet's overlayPane (z 400) so a raster added later --
    # a GridLayer in the tilePane (z 200) -- can never cover them
    assert layer.pane == "overlayPane"


def test_clear_reference_points_removes_layer(tmp_path):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    m = _FakeMap()
    m.reference_points_layer = "REF"
    m.reference_points_dir = str(ref_dir)

    SbaeMap.clear_reference_points(m)

    assert m.removed == ["ref_pts"]  # removed by stable key, not the tracked object
    assert m.reference_points_layer is None
    assert m.reference_points_dir is None
    assert not ref_dir.exists()  # its backing dir is reclaimed


def test_clear_reference_points_tolerates_layer_already_gone():
    """Clearing tolerates a reference layer already gone from the shared map.

    Reproduces the live crash: the map can drop the reference layer out from
    under the tracker (theme change / re-style swaps the widget), so pysepal's
    ``remove_layer`` raises for the stale handle. Clearing must use key-based
    removal with ``none_ok`` so it never propagates that error.
    """

    class _StrictMap:
        def __init__(self):
            self.reference_points_layer = "STALE"  # tracked, but not on the map
            self.reference_points_dir = None
            self.removed = []

        def remove_layer(self, key, base=False, none_ok=False):
            if not none_ok:
                raise ValueError(f"no layer corresponding to {key} on the map")
            self.removed.append(key)

    m = _StrictMap()
    SbaeMap.clear_reference_points(m)  # must not raise
    assert m.reference_points_layer is None
    assert m.removed == ["ref_pts"]


def test_compose_points_legend():
    compose = map_mod._compose_points_legend
    assert compose(False, False, False) == {}
    assert compose(True, False, False) == {
        map_mod._SAMPLE_LEGEND_LABEL: map_mod.SAMPLE_POINT_COLOR
    }
    # reference evaluated -> green/red correctness key
    assert compose(False, True, True) == {
        map_mod._CORRECT_LEGEND_LABEL: map_mod.CORRECT_COLOR,
        map_mod._INCORRECT_LEGEND_LABEL: map_mod.INCORRECT_COLOR,
    }
    # reference not yet evaluated -> single neutral entry
    assert compose(False, True, False) == {
        map_mod._REFERENCE_LEGEND_LABEL: map_mod.REFERENCE_NEUTRAL_COLOR
    }
    # both layers -> sample + correctness
    assert set(compose(True, True, True)) == {
        map_mod._SAMPLE_LEGEND_LABEL,
        map_mod._CORRECT_LEGEND_LABEL,
        map_mod._INCORRECT_LEGEND_LABEL,
    }


def test_add_reference_points_colors_by_correctness(monkeypatch):
    captured = {}

    class _Layer:
        pass

    async def fake_build(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        captured["colors"] = colors
        captured["kwargs"] = kwargs
        captured["df"] = df
        return _Layer()

    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", fake_build)
    m = _FakeMap()
    df = pd.DataFrame(
        {
            "longitude": [1.0, 2.0, 3.0],
            "latitude": [1.0, 2.0, 3.0],
            "map_code": [1, 2, 5],
            "ref_code": [1, 3, 5],  # row 2 disagrees -> incorrect
        }
    )
    asyncio.run(SbaeMap.add_reference_points(m, df))

    # attached under the distinct ref key, tracked apart from the design layer
    assert m.added and m.added[0][1] == "ref_pts"
    assert m.reference_points_layer is not None
    assert m.sample_points_layer is None  # design layer untouched
    assert m.added[0][0].name == "Reference points"
    # coloured by agreement only: correct=green(1), incorrect=red(0)
    assert captured["kwargs"]["color_field"] == "correct"
    assert captured["kwargs"]["radius"] == 6
    assert captured["colors"] == {0: map_mod.INCORRECT_COLOR, 1: map_mod.CORRECT_COLOR}
    assert list(captured["df"]["correct"]) == [1, 0, 1]  # agree, disagree, agree
    assert m._reference_evaluated is True


def test_add_reference_points_neutral_without_map_code(monkeypatch):
    captured = {}

    class _Layer:
        pass

    async def fake_build(df, colors, *, dest_dir, default_color="#888888", **kwargs):
        captured["colors"] = colors
        captured["kwargs"] = kwargs
        captured["default_color"] = default_color
        return _Layer()

    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", fake_build)
    m = _FakeMap()
    # no map_code -> correctness unknown (e.g. raster not sampled yet)
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "ref_code": [3]})
    asyncio.run(SbaeMap.add_reference_points(m, df))

    # single neutral colour, not green/red
    assert captured["colors"] == {}
    assert captured["default_color"] == map_mod.REFERENCE_NEUTRAL_COLOR
    assert m._reference_evaluated is False
