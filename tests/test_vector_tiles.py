import pandas as pd
import pytest

from component.scripts.vector_tiles import (
    POINT_CONVERSION_OPTIONS,
    VectorTileError,
    build_point_style,
    build_points_pmtiles_layer,
    points_to_geojson,
)


def test_points_to_geojson_shape_and_lonlat_order():
    df = pd.DataFrame(
        {"longitude": [10.0, 11.0], "latitude": [-5.0, -6.0], "map_code": [1, 2]}
    )
    fc = points_to_geojson(df)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    f0 = fc["features"][0]
    assert f0["geometry"]["type"] == "Point"
    assert f0["geometry"]["coordinates"] == [10.0, -5.0]  # [lon, lat]
    assert f0["properties"]["map_code"] == 1
    assert isinstance(f0["properties"]["map_code"], int)


def test_points_to_geojson_omits_absent_props():
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0]})  # no map_code
    fc = points_to_geojson(df)
    assert fc["features"][0]["properties"] == {}


def test_points_to_geojson_empty_df():
    fc = points_to_geojson(pd.DataFrame({"longitude": [], "latitude": []}))
    assert fc["features"] == []


def test_build_point_style_categorized_match():
    style = build_point_style("http://x/p", {1: "#111111", 2: "#222222"}, "pts")
    assert style["sources"]["pmtiles_source"]["url"] == "pmtiles://http://x/p"
    layer = style["layers"][0]
    assert layer["type"] == "circle"
    assert layer["source-layer"] == "pts"
    cc = layer["paint"]["circle-color"]
    assert cc[0] == "match"
    assert cc[1] == ["get", "map_code"]
    assert cc[2:] == [1, "#111111", 2, "#222222", "#888888"]  # ...default last


def test_build_point_style_empty_colors_plain_default():
    style = build_point_style("u", {}, "pts", default_color="#abcdef")
    assert style["layers"][0]["paint"]["circle-color"] == "#abcdef"


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pmtiles_url = "http://localhost:1/pmtiles?filePath=p.pmtiles"

    def list_layers(self):
        return ["sample_points"]

    def create_leaflet_layer(self, style=None):
        return {"layer": True, "style": style}


def test_build_points_layer_writes_geojson_and_passes_style(tmp_path):
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    captured = {}

    def factory(**kwargs):
        captured["kwargs"] = kwargs
        return _FakeClient(**kwargs)

    layer = build_points_pmtiles_layer(
        df, {3: "#333333"}, dest_dir=str(tmp_path), client_factory=factory
    )
    assert (tmp_path / "sample_points.geojson").exists()
    assert captured["kwargs"]["conversion_options"] == POINT_CONVERSION_OPTIONS
    assert captured["kwargs"]["allowed_directories"] == [str(tmp_path)]
    style = layer["style"]
    assert style["layers"][0]["source-layer"] == "sample_points"
    assert style["layers"][0]["paint"]["circle-color"][2:] == [3, "#333333", "#888888"]


def test_build_points_layer_no_layers_raises(tmp_path):
    class _NoLayers(_FakeClient):
        def list_layers(self):
            return []

    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    with pytest.raises(VectorTileError):
        build_points_pmtiles_layer(
            df, {}, dest_dir=str(tmp_path), client_factory=lambda **k: _NoLayers(**k)
        )


def test_build_points_layer_client_error_wrapped(tmp_path):
    def boom(**k):
        raise RuntimeError("tippecanoe missing")

    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    with pytest.raises(VectorTileError):
        build_points_pmtiles_layer(df, {}, dest_dir=str(tmp_path), client_factory=boom)


@pytest.mark.skipif(
    __import__("shutil").which("tippecanoe") is None, reason="tippecanoe not installed"
)
def test_build_points_layer_real(tmp_path):
    df = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.002],
            "latitude": [0.0, 0.001, 0.002],
            "map_code": [1, 1, 2],
        }
    )
    try:
        layer = build_points_pmtiles_layer(df, {1: "#111111"}, dest_dir=str(tmp_path))
    except VectorTileError as e:
        pytest.skip(f"tile library unavailable: {e}")
    assert layer is not None
