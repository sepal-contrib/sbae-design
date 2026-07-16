import types

import pandas as pd
import pytest

from component.widget import map as map_mod
from component.widget.map import SbaeMap
from component.scripts.vector_tiles import VectorTileError


class _FakeMap:
    def __init__(self):
        self.sample_points_layer = None
        self.added = []
        self.removed = []

    def add_layer(self, layer, key=None):
        self.added.append((layer, key))

    def remove_layer(self, layer):
        self.removed.append(layer)


def test_attach_swaps_layer():
    m = _FakeMap()
    SbaeMap.attach_sample_points_layer(m, "L1")
    assert m.added == [("L1", "sample_pts")]
    assert m.sample_points_layer == "L1"
    SbaeMap.attach_sample_points_layer(m, "L2")
    assert m.removed == ["L1"]
    assert m.sample_points_layer == "L2"


def test_build_empty_returns_none():
    assert SbaeMap.build_sample_points_layer(_FakeMap(), pd.DataFrame(), {}) is None


def test_build_delegates(monkeypatch):
    called = {}

    def fake_build(df, colors, *, dest_dir):
        called["colors"] = colors
        return "LAYER"

    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", fake_build)
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    layer = SbaeMap.build_sample_points_layer(_FakeMap(), df, {3: "#333333"})
    assert layer == "LAYER"
    assert called["colors"] == {3: "#333333"}


def test_add_sample_points_error_notifies(monkeypatch):
    from component.model import app_state as real_app_state

    def boom(df, colors, *, dest_dir):
        raise VectorTileError("nope")

    errs = []
    monkeypatch.setattr(map_mod, "build_points_pmtiles_layer", boom)
    monkeypatch.setattr(real_app_state, "add_error", lambda msg: errs.append(msg))
    m = _FakeMap()
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    SbaeMap.add_sample_points(m, df)  # must not raise
    assert errs and "Could not render sample points" in errs[0]
    assert m.added == []
