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
