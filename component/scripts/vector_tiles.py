"""Render sample points as a PMTiles vector-tile layer.

DataFrame -> GeoJSON, the MapLibre circle style, and the call into the tile
library all live here so they stay pure and unit-testable. The tile-library
import is confined to ``_default_client_factory`` -- the ONE seam that touches
``vectortileserver``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger("sbae.vector_tiles")


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


class VectorTileError(Exception):
    """Raised when a PMTiles point layer cannot be built."""


# tippecanoe options that keep 100% of point features at every zoom. Accuracy
# assessment cannot silently drop points, so dot-dropping and the tile/feature
# limits are disabled. See the ipy-pmtiles contract, requirement R2.
POINT_CONVERSION_OPTIONS = {
    "drop_rate": 1,  # -r1: no dot-dropping below maxzoom
    "no_feature_limit": True,  # -pf
    "no_tile_size_limit": True,  # -pk
}


def _default_client_factory(**kwargs):
    """The ONE seam where the tile-library is imported.

    ``vectortileserver`` (PyPI) is the only external tile dependency; nothing
    else in the codebase references it.
    """
    from vectortileserver.client import TileClient

    return TileClient(**kwargs)


def build_points_pmtiles_layer(
    df: pd.DataFrame,
    class_colors: dict,
    *,
    dest_dir: str,
    color_field: str = "map_code",
    client_factory: Optional[Callable] = None,
):
    """Convert points to PMTiles and return an ipyleaflet layer.

    Writes the GeoJSON into ``dest_dir``, runs the tile client (tippecanoe) with
    point-retention options, builds the circle style, and returns the leaflet
    layer. Raises :class:`VectorTileError` on any failure (missing library,
    tippecanoe error, no vector layers produced).
    """
    if client_factory is None:
        client_factory = _default_client_factory

    geojson_path = os.path.join(dest_dir, "sample_points.geojson")
    try:
        with open(geojson_path, "w") as fh:
            json.dump(points_to_geojson(df, props=(color_field,)), fh)
    except (OSError, TypeError) as e:
        raise VectorTileError(f"Could not write points GeoJSON: {e}") from e

    try:
        client = client_factory(
            data_source=geojson_path,
            conversion_options=POINT_CONVERSION_OPTIONS,
            allowed_directories=[dest_dir],
        )
        layers = client.list_layers()
        if not layers:
            raise VectorTileError("Tile conversion produced no vector layers.")
        style = build_point_style(
            client.pmtiles_url, class_colors, layers[0], color_field=color_field
        )
        return client.create_leaflet_layer(style=style)
    except VectorTileError:
        raise
    except Exception as e:
        raise VectorTileError(f"Could not build PMTiles point layer: {e}") from e


def build_layer_or_notify(sbae_map, points_df, class_colors):
    """Build the sample-points layer, notifying (not raising) on failure.

    Returns the layer, or ``None`` when there is no map, no points, or the build
    failed (an error is surfaced via ``app_state.add_error``). Safe to call from
    a worker thread; the caller attaches the returned layer on the main thread.
    """
    if sbae_map is None or points_df is None or points_df.empty:
        return None
    from component.model import app_state

    try:
        return sbae_map.build_sample_points_layer(points_df, class_colors)
    except Exception as e:
        # Intentionally broad: this is a non-essential map-layer build, called
        # from the point-generation worker thread. Any failure here -- not just
        # VectorTileError -- must be swallowed and reported, never raised, or
        # the worker thread dies before it can hand the already-generated
        # points back to the app (see build_sample_points_layer's mkdtemp call,
        # which can raise a raw OSError outside build_points_pmtiles_layer's
        # own try/except).
        logger.warning("Sample points layer failed: %s", e)
        app_state.add_error(f"Could not render sample points on the map: {e}")
        return None
