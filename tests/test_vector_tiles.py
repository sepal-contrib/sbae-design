import pandas as pd
import pytest

from component.scripts.vector_tiles import (
    POINT_CONVERSION_OPTIONS,
    VectorTileError,
    build_layer_or_notify,
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


def test_build_points_layer_writes_geojson_and_passes_style(tmp_path):
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    captured = {}

    def factory(source, *, style, conversion_options, allowed_directories):
        captured["conversion_options"] = conversion_options
        captured["allowed_directories"] = allowed_directories
        # ``style`` is a builder; resolve it with fake archive metadata + URL.
        metadata = {"vector_layers": [{"id": "sample_points"}]}
        resolved = style(metadata, "http://localhost:1/pmtiles?filePath=p.pmtiles")
        return {"layer": True, "style": resolved, "bounds": [[0.0, 0.0], [1.0, 1.0]]}

    layer = build_points_pmtiles_layer(
        df, {3: "#333333"}, dest_dir=str(tmp_path), layer_factory=factory
    )
    assert (tmp_path / "sample_points.geojson").exists()
    assert captured["conversion_options"] == POINT_CONVERSION_OPTIONS
    assert captured["allowed_directories"] == [str(tmp_path)]
    style = layer["style"]
    assert style["layers"][0]["source-layer"] == "sample_points"
    assert style["layers"][0]["paint"]["circle-color"][2:] == [3, "#333333", "#888888"]
    assert layer["bounds"] == [[0.0, 0.0], [1.0, 1.0]]


def test_build_points_layer_no_layers_raises(tmp_path):
    def factory(source, *, style, conversion_options, allowed_directories):
        # Empty archive metadata -> the style builder must raise.
        return {"style": style({"vector_layers": []}, "u")}

    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    with pytest.raises(VectorTileError):
        build_points_pmtiles_layer(
            df, {}, dest_dir=str(tmp_path), layer_factory=factory
        )


def test_build_points_layer_open_error_wrapped(tmp_path):
    def boom(source, *, style, conversion_options, allowed_directories):
        raise RuntimeError("tippecanoe missing")

    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    with pytest.raises(VectorTileError):
        build_points_pmtiles_layer(df, {}, dest_dir=str(tmp_path), layer_factory=boom)


def test_build_points_layer_geojson_serialization_error_wrapped(tmp_path):
    # All-integer columns (including lon/lat) so iterrows() keeps the row dtype
    # int64 instead of upcasting to float64 -- that upcast is what makes
    # np.int64 JSON-serializable (it becomes a plain float subtype), so it
    # would silently hide the bug this test guards against.
    df = pd.DataFrame({"longitude": [1], "latitude": [2], "class_id": [7]})
    with pytest.raises(VectorTileError):
        build_points_pmtiles_layer(
            df,
            {7: "#333333"},
            dest_dir=str(tmp_path),
            color_field="class_id",
            layer_factory=lambda source, **k: {"layer": True},
        )


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


class _MapOK:
    def build_sample_points_layer(self, df, colors):
        return "LAYER"


class _MapBoom:
    def build_sample_points_layer(self, df, colors):
        raise VectorTileError("bad raster")


def test_build_layer_or_notify_success():
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    assert build_layer_or_notify(_MapOK(), df, {3: "#333333"}) == "LAYER"


def test_build_layer_or_notify_none_map_or_empty():
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0]})
    assert build_layer_or_notify(None, df, {}) is None
    assert build_layer_or_notify(_MapOK(), pd.DataFrame(), {}) is None


def test_build_layer_or_notify_failure_notifies(monkeypatch):
    from component.model import app_state as real_app_state

    errs = []
    monkeypatch.setattr(real_app_state, "add_error", lambda msg: errs.append(msg))
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    assert build_layer_or_notify(_MapBoom(), df, {}) is None
    assert errs and "Could not render sample points" in errs[0]


class _MapNonVectorTileBoom:
    def build_sample_points_layer(self, df, colors):
        # e.g. tempfile.mkdtemp() failing with a raw OSError (disk full /
        # quota) before build_points_pmtiles_layer -- and its VectorTileError
        # wrapping -- is ever reached.
        raise OSError("disk full")


def test_build_layer_or_notify_non_vector_tile_error_notifies(monkeypatch):
    from component.model import app_state as real_app_state

    errs = []
    monkeypatch.setattr(real_app_state, "add_error", lambda msg: errs.append(msg))
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    assert build_layer_or_notify(_MapNonVectorTileBoom(), df, {}) is None
    assert errs and "Could not render sample points" in errs[0]
