"""Render sample points as a PMTiles vector-tile layer.

DataFrame -> GeoJSON, the MapLibre circle style, and the call into the tile
library all live here so they stay pure and unit-testable. The tile-library
import is confined to ``_default_client_factory`` -- the ONE seam where the
(not-yet-published) package name appears.
"""

from __future__ import annotations

import pandas as pd


def points_to_geojson(
    df: pd.DataFrame,
    x_col: str = "longitude",
    y_col: str = "latitude",
    props: tuple[str, ...] = ("map_code",),
) -> dict:
    """Build a GeoJSON FeatureCollection of points from a DataFrame.

    Coordinates are written ``[lon, lat]`` (GeoJSON order). Only the columns in
    ``props`` that are present in ``df`` are copied onto each feature; NaN
    values are skipped. ``map_code`` is coerced to ``int`` for clean JSON.
    """
    present = [p for p in props if p in df.columns]
    features = []
    for _, row in df.iterrows():
        properties = {}
        for p in present:
            value = row[p]
            if pd.isna(value):
                continue
            properties[p] = int(value) if p == "map_code" else value
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row[x_col]), float(row[y_col])],
                },
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_point_style(
    pmtiles_url: str,
    class_colors: dict,
    source_layer: str,
    *,
    color_field: str = "map_code",
    default_color: str = "#888888",
    radius: int = 5,
    stroke_color: str = "#ffffff",
    stroke_width: int = 1,
    opacity: float = 0.85,
) -> dict:
    """Build a MapLibre style with one categorized ``circle`` layer.

    ``circle-color`` is a ``["match", ["get", color_field], code, hex, ...,
    default]`` expression from ``class_colors``. Empty ``class_colors`` yields a
    plain ``default_color`` (no match expression).
    """
    if class_colors:
        circle_color = ["match", ["get", color_field]]
        for code, hex_color in class_colors.items():
            circle_color.extend([int(code), hex_color])
        circle_color.append(default_color)
    else:
        circle_color = default_color

    return {
        "version": 8,
        "sources": {
            "pmtiles_source": {"type": "vector", "url": f"pmtiles://{pmtiles_url}"}
        },
        "layers": [
            {
                "id": "sample-points-circle",
                "type": "circle",
                "source": "pmtiles_source",
                "source-layer": source_layer,
                "paint": {
                    "circle-radius": radius,
                    "circle-color": circle_color,
                    "circle-stroke-color": stroke_color,
                    "circle-stroke-width": stroke_width,
                    "circle-opacity": opacity,
                },
            }
        ],
    }
