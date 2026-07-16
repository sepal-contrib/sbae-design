import pandas as pd

from component.scripts.vector_tiles import build_point_style, points_to_geojson


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
