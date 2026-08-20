import asyncio
import sys

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


def test_points_to_geojson_emits_multiple_code_props():
    df = pd.DataFrame(
        {"longitude": [1.0], "latitude": [2.0], "map_code": [3], "ref_code": [4]}
    )
    props = points_to_geojson(df, props=("map_code", "ref_code"))["features"][0][
        "properties"
    ]
    assert props["map_code"] == 3 and isinstance(props["map_code"], int)
    assert props["ref_code"] == 4 and isinstance(props["ref_code"], int)


def test_default_layer_factory_opens_a_workspace_for_the_dest_dir(monkeypatch):
    import types

    from component.scripts import vector_tiles as vt

    captured = {}

    class _FakeWorkspace:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def open_async(self, source, *, style, conversion_options):
            return "LAYER"

    monkeypatch.setitem(
        sys.modules,
        "vectortileserver",
        types.SimpleNamespace(TileWorkspace=_FakeWorkspace),
    )
    layer = asyncio.run(
        vt._default_layer_factory(
            "src.geojson",
            style=lambda *a: {},
            conversion_options={},
            allowed_directories=["/tmp/x"],
        )
    )
    assert layer == "LAYER"
    assert captured["allowed_directories"] == ["/tmp/x"]
    # The IPv4 bind is vectortileserver's own default from 0.2.2 on. Passing a
    # host here again would put the band-aid back in the app layer.
    assert "host" not in captured


def test_points_to_geojson_omits_absent_props():
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0]})  # no map_code
    fc = points_to_geojson(df)
    assert fc["features"][0]["properties"] == {}


def test_points_to_geojson_empty_df():
    fc = points_to_geojson(pd.DataFrame({"longitude": [], "latitude": []}))
    assert fc["features"] == []


def test_build_point_style_emits_flat_layer_per_class():
    # protomaps-leaflet (the PMTiles renderer) can't evaluate MapLibre colour
    # expressions, so each class becomes its own flat-coloured, filtered circle
    # layer rather than one layer with a ["match", ...] circle-color.
    style = build_point_style(
        "http://x/p", {1: "#111111", 2: "#222222"}, "pts", categories=[1, 2]
    )
    assert style["sources"]["pmtiles_source"]["url"] == "pmtiles://http://x/p"
    layers = style["layers"]
    assert [layer["type"] for layer in layers] == ["circle", "circle"]
    assert all(layer["source-layer"] == "pts" for layer in layers)
    # flat string colours (never an expression array), each behind a == filter
    assert all(isinstance(layer["paint"]["circle-color"], str) for layer in layers)
    by_color = {layer["paint"]["circle-color"]: layer for layer in layers}
    assert by_color["#111111"]["filter"] == ["==", "map_code", 1]
    assert by_color["#222222"]["filter"] == ["==", "map_code", 2]


def test_build_point_style_empty_colors_single_flat_layer():
    style = build_point_style("u", {}, "pts", default_color="#abcdef")
    (layer,) = style["layers"]
    assert layer["paint"]["circle-color"] == "#abcdef"
    assert "filter" not in layer  # nothing to filter on -- one plain layer


def test_build_point_style_greenred_by_correctness():
    # Analysis colours by agreement only: correct=green, incorrect=red, one flat
    # filtered layer each -- no per-class palette.
    style = build_point_style(
        "u",
        {0: "#c62828", 1: "#2e7d32"},
        "pts",
        color_field="correct",
        categories=[0, 1],
    )
    by_color = {
        layer["paint"]["circle-color"]: layer["filter"] for layer in style["layers"]
    }
    assert by_color["#2e7d32"] == ["==", "correct", 1]  # correct -> green
    assert by_color["#c62828"] == ["==", "correct", 0]  # incorrect -> red


def test_build_point_style_uniform_ring_colour():
    # The ring (halo) is always a plain fixed colour.
    style = build_point_style(
        "u", {1: "#111111"}, "pts", categories=[1], stroke_color="#ffffff"
    )
    assert style["layers"][0]["paint"]["circle-stroke-color"] == "#ffffff"


def test_build_points_layer_writes_geojson_and_passes_style(tmp_path):
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    captured = {}

    async def factory(source, *, style, conversion_options, allowed_directories):
        captured["conversion_options"] = conversion_options
        captured["allowed_directories"] = allowed_directories
        # ``style`` is a builder; resolve it with fake archive metadata + URL.
        metadata = {"vector_layers": [{"id": "sample_points"}]}
        resolved = style(metadata, "http://localhost:1/pmtiles?filePath=p.pmtiles")
        return {"layer": True, "style": resolved, "bounds": [[0.0, 0.0], [1.0, 1.0]]}

    layer = asyncio.run(
        build_points_pmtiles_layer(
            df, {3: "#333333"}, dest_dir=str(tmp_path), layer_factory=factory
        )
    )
    assert (tmp_path / "sample_points.geojson").exists()
    assert captured["conversion_options"] == POINT_CONVERSION_OPTIONS
    assert captured["allowed_directories"] == [str(tmp_path)]
    style = layer["style"]
    circle = style["layers"][0]
    assert circle["source-layer"] == "sample_points"
    # per-class flat layer, filtered on the observed map code (no colour expr)
    assert circle["paint"]["circle-color"] == "#333333"
    assert circle["filter"] == ["==", "map_code", 3]
    assert layer["bounds"] == [[0.0, 0.0], [1.0, 1.0]]


def test_build_points_layer_categorizes_by_color_field(tmp_path):
    import json

    # Analysis path: colour by a derived "correct" field -> per-value flat layers.
    df = pd.DataFrame(
        {"longitude": [1.0, 2.0], "latitude": [2.0, 3.0], "correct": [1, 0]}
    )
    captured = {}

    async def factory(source, *, style, conversion_options, allowed_directories):
        metadata = {"vector_layers": [{"id": "sample_points"}]}
        captured["style"] = style(metadata, "http://x/pmtiles")
        return {"style": captured["style"]}

    asyncio.run(
        build_points_pmtiles_layer(
            df,
            {0: "#c62828", 1: "#2e7d32"},
            dest_dir=str(tmp_path),
            color_field="correct",
            radius=6,
            layer_factory=factory,
        )
    )
    fc = json.load(open(tmp_path / "sample_points.geojson"))
    # the colour field must ride along in the tile features (the == filter needs it)
    assert "correct" in fc["features"][0]["properties"]
    by_color = {
        layer["paint"]["circle-color"]: layer["filter"]
        for layer in captured["style"]["layers"]
    }
    assert by_color["#2e7d32"] == ["==", "correct", 1]
    assert by_color["#c62828"] == ["==", "correct", 0]


def test_build_points_layer_no_layers_raises(tmp_path):
    async def factory(source, *, style, conversion_options, allowed_directories):
        # Empty archive metadata -> the style builder must raise.
        return {"style": style({"vector_layers": []}, "u")}

    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    with pytest.raises(VectorTileError):
        asyncio.run(
            build_points_pmtiles_layer(
                df, {}, dest_dir=str(tmp_path), layer_factory=factory
            )
        )


def test_build_points_layer_open_error_wrapped(tmp_path):
    async def boom(source, *, style, conversion_options, allowed_directories):
        raise RuntimeError("tippecanoe missing")

    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    with pytest.raises(VectorTileError):
        asyncio.run(
            build_points_pmtiles_layer(
                df, {}, dest_dir=str(tmp_path), layer_factory=boom
            )
        )


def test_build_points_layer_geojson_serialization_error_wrapped(tmp_path):
    # All-integer columns (including lon/lat) so iterrows() keeps the row dtype
    # int64 instead of upcasting to float64 -- that upcast is what makes
    # np.int64 JSON-serializable (it becomes a plain float subtype), so it
    # would silently hide the bug this test guards against.
    df = pd.DataFrame({"longitude": [1], "latitude": [2], "class_id": [7]})

    async def unused_factory(source, **kwargs):
        return {"layer": True}

    with pytest.raises(VectorTileError):
        asyncio.run(
            build_points_pmtiles_layer(
                df,
                {7: "#333333"},
                dest_dir=str(tmp_path),
                color_field="class_id",
                layer_factory=unused_factory,
            )
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
        layer = asyncio.run(
            build_points_pmtiles_layer(df, {1: "#111111"}, dest_dir=str(tmp_path))
        )
    except VectorTileError as e:
        pytest.skip(f"tile library unavailable: {e}")
    assert layer is not None


class _MapOK:
    async def build_sample_points_layer(self, df, *args, **kwargs):
        return "LAYER"


class _MapBoom:
    async def build_sample_points_layer(self, df, *args, **kwargs):
        raise VectorTileError("bad raster")


def test_build_layer_or_notify_success():
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    assert asyncio.run(build_layer_or_notify(_MapOK(), df)) == "LAYER"


def test_build_layer_or_notify_none_map_or_empty():
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0]})
    assert asyncio.run(build_layer_or_notify(None, df)) is None
    assert asyncio.run(build_layer_or_notify(_MapOK(), pd.DataFrame())) is None


def test_build_layer_or_notify_failure_notifies(monkeypatch):
    from component.model import app_state as real_app_state

    errs = []
    monkeypatch.setattr(real_app_state, "add_error", lambda msg: errs.append(msg))
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    assert asyncio.run(build_layer_or_notify(_MapBoom(), df)) is None
    assert errs and "Could not render sample points" in errs[0]


class _MapNonVectorTileBoom:
    async def build_sample_points_layer(self, df, *args, **kwargs):
        # e.g. scratch_dir() failing with a raw OSError (disk full / quota)
        # before build_points_pmtiles_layer -- and its VectorTileError
        # wrapping -- is ever reached.
        raise OSError("disk full")


def test_build_layer_or_notify_non_vector_tile_error_notifies(monkeypatch):
    from component.model import app_state as real_app_state

    errs = []
    monkeypatch.setattr(real_app_state, "add_error", lambda msg: errs.append(msg))
    df = pd.DataFrame({"longitude": [1.0], "latitude": [2.0], "map_code": [3]})
    assert asyncio.run(build_layer_or_notify(_MapNonVectorTileBoom(), df)) is None
    assert errs and "Could not render sample points" in errs[0]
