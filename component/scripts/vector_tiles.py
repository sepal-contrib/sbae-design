"""Render sample points as a PMTiles vector-tile layer.

DataFrame -> GeoJSON, the MapLibre circle style, and the call into the tile
library all live here so they stay pure and unit-testable. The tile-library
import is confined to ``_default_client_factory`` -- the ONE seam that touches
``vectortileserver``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable, Optional

import pandas as pd

from component.scripts.logging_config import quiet_tile_server_logs

logger = logging.getLogger("sbae.vector_tiles")


# Point styling. Sample/reference points stay a single neutral colour with a
# white halo so they read over a colourful classification map; analysis points
# encode only agreement (green = map matches reference, red = it doesn't).
SAMPLE_POINT_COLOR = "#000000"
POINT_HALO_COLOR = "#ffffff"
CORRECT_COLOR = "#2e7d32"
INCORRECT_COLOR = "#c62828"
REFERENCE_NEUTRAL_COLOR = "#607d8b"

# Feature properties coerced to int for clean JSON *and* so protomaps-leaflet's
# strict-equality ``==`` filter matches the numeric tile values.
_INT_PROPS = ("map_code", "ref_code", "correct")


def points_to_geojson(
    df: pd.DataFrame,
    x_col: str = "longitude",
    y_col: str = "latitude",
    props: tuple[str, ...] = ("map_code",),
) -> dict:
    """Build a GeoJSON FeatureCollection of points from a DataFrame.

    Coordinates are written ``[lon, lat]`` (GeoJSON order). Only the columns in
    ``props`` that are present in ``df`` are copied onto each feature; NaN
    values are skipped. Categorical fields (see ``_INT_PROPS``) are coerced to
    ``int`` for clean JSON.
    """
    present = [p for p in props if p in df.columns]
    features = []
    for _, row in df.iterrows():
        properties = {}
        for p in present:
            value = row[p]
            if pd.isna(value):
                continue
            properties[p] = int(value) if p in _INT_PROPS else value
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


def _circle_layer(
    idx: int, source_layer: str, filt, *, fill, stroke, radius, stroke_width, opacity
) -> dict:
    layer = {
        "id": f"sample-points-circle-{idx}",
        "type": "circle",
        "source": "pmtiles_source",
        "source-layer": source_layer,
        "paint": {
            "circle-radius": radius,
            "circle-color": fill,
            "circle-stroke-color": stroke,
            "circle-stroke-width": stroke_width,
            "circle-opacity": opacity,
        },
    }
    if filt is not None:
        layer["filter"] = filt
    return layer


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
    categories: list | None = None,
) -> dict:
    """Build a MapLibre-style dict of flat-colour ``circle`` layers, one per class.

    The PMTiles renderer (protomaps-leaflet, via ipyleaflet) does NOT evaluate
    MapLibre data-driven expressions for colours -- ``circle-color`` must be a
    plain colour string (or a JS function, which can't cross the JSON style
    trait). An expression array is passed through untouched and the canvas falls
    back to solid black. So categorical colouring is done the protomaps-leaflet
    way: one flat-colour circle layer per category value, each behind a legacy
    ``["==", color_field, code]`` filter (the only filter form protomaps-leaflet
    understands).

    ``categories`` lists the codes to emit, derived from the data actually
    present. Empty ``class_colors`` yields a single unfiltered ``default_color``
    layer -- the uniform look used for the design sample points.
    """
    common = dict(radius=radius, stroke_width=stroke_width, opacity=opacity)
    layers = []

    if not class_colors:
        layers.append(
            _circle_layer(
                0, source_layer, None, fill=default_color, stroke=stroke_color, **common
            )
        )
    else:
        codes = categories if categories is not None else list(class_colors)
        for idx, code in enumerate(codes):
            layers.append(
                _circle_layer(
                    idx,
                    source_layer,
                    ["==", color_field, int(code)],
                    fill=class_colors.get(code, default_color),
                    stroke=stroke_color,
                    **common,
                )
            )

    return {
        "version": 8,
        "sources": {
            "pmtiles_source": {"type": "vector", "url": f"pmtiles://{pmtiles_url}"}
        },
        "layers": layers,
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


async def _default_layer_factory(
    source, *, style, conversion_options, allowed_directories
):
    """The ONE seam where the tile-library is imported.

    ``vectortileserver`` (PyPI) is the only external tile dependency; nothing
    else in the codebase references it. A per-build ``TileWorkspace`` carries the
    temp ``dest_dir`` as an allowed directory so the shared tile server can serve
    it. ``open_async`` offloads tippecanoe to a thread and builds the layer widget
    on the event loop, returning a layer that already knows its own ``.bounds``.
    """
    import vectortileserver as vts

    workspace = vts.TileWorkspace(allowed_directories=allowed_directories)
    return await workspace.open_async(
        source, style=style, conversion_options=conversion_options
    )


def _point_categories(df: pd.DataFrame, color_field: str):
    """Distinct ``color_field`` values present in ``df``, one flat layer each.

    Coerced to ``int`` because protomaps-leaflet's ``==`` filter is strict
    equality against the numeric tile properties. ``None`` when the field is
    absent (nothing to categorize -> a single flat layer).
    """
    if color_field in df.columns:
        return sorted(int(v) for v in df[color_field].dropna().unique())
    return None


async def build_points_pmtiles_layer(
    df: pd.DataFrame,
    class_colors: dict,
    *,
    dest_dir: str,
    color_field: str = "map_code",
    default_color: str = "#888888",
    stroke_color: str = POINT_HALO_COLOR,
    layer_factory: Optional[Callable] = None,
    radius: int = 5,
    stroke_width: int = 1,
):
    """Convert points to PMTiles and return an ipyleaflet layer.

    Writes the GeoJSON into ``dest_dir``, opens it through ``vectortileserver``
    (tippecanoe, with point-retention options) with a per-class ``circle`` style,
    and returns the layer -- which carries its own ``.bounds`` for auto-zoom.
    Awaitable: the geojson write and the conversion both run off the event loop.
    Raises :class:`VectorTileError` on any failure (missing library, tippecanoe
    error, no vector layers produced). ``color_field`` rides along as a
    tile-feature property so its per-class filters match.
    """
    if layer_factory is None:
        layer_factory = _default_layer_factory

    geojson_path = os.path.join(dest_dir, "sample_points.geojson")
    props = (color_field,)
    categories = _point_categories(df, color_field)

    def _write_geojson():
        with open(geojson_path, "w") as fh:
            json.dump(points_to_geojson(df, props=props), fh)

    try:
        await asyncio.to_thread(_write_geojson)
    except (OSError, TypeError) as e:
        raise VectorTileError(f"Could not write points GeoJSON: {e}") from e

    def style_from(metadata, pmtiles_url):
        # A style *builder*: the circle style needs the tile URL and the archive's
        # own vector-layer id, neither known until the archive has been built.
        vector_layers = metadata.get("vector_layers", [])
        if not vector_layers:
            raise VectorTileError("Tile conversion produced no vector layers.")
        return build_point_style(
            pmtiles_url,
            class_colors,
            vector_layers[0]["id"],
            color_field=color_field,
            default_color=default_color,
            stroke_color=stroke_color,
            radius=radius,
            stroke_width=stroke_width,
            categories=categories,
        )

    try:
        layer = await layer_factory(
            geojson_path,
            style=style_from,
            conversion_options=POINT_CONVERSION_OPTIONS,
            allowed_directories=[dest_dir],
        )
        # The tile server just booted its uvicorn (and vectortileserver's DEBUG
        # logger); pin their levels down now so it sticks past uvicorn's config.
        quiet_tile_server_logs()
        return layer
    except VectorTileError:
        raise
    except Exception as e:
        raise VectorTileError(f"Could not build PMTiles point layer: {e}") from e


async def build_layer_or_notify(sbae_map, points_df):
    """Build the design sample-points layer, notifying (not raising) on failure.

    Returns the layer, or ``None`` when there is no map, no points, or the build
    failed (an error is surfaced via ``app_state.add_error``). Awaitable; the
    caller attaches the returned layer on the event loop. Design points are a
    single neutral colour (see ``SAMPLE_POINT_COLOR``) -- per-class colouring
    only fights the classification map underneath.
    """
    if sbae_map is None or points_df is None or points_df.empty:
        return None
    from component.model import app_state

    try:
        return await sbae_map.build_sample_points_layer(
            points_df, point_color=SAMPLE_POINT_COLOR
        )
    except Exception as e:
        # Intentionally broad: this is a non-essential map-layer build, called
        # from the point-generation worker thread. Any failure here -- not just
        # VectorTileError -- must be swallowed and reported, never raised, or
        # the worker thread dies before it can hand the already-generated
        # points back to the app (see build_sample_points_layer's scratch_dir
        # call, which can raise a raw OSError outside build_points_pmtiles_layer's
        # own try/except).
        logger.warning("Sample points layer failed: %s", e)
        app_state.add_error(f"Could not render sample points on the map: {e}")
        return None
