"""Ground area of raster pixels: planar where the CRS allows it, else geodesic.

``choose_area_method`` measures how far pixel-count x pixel-size departs from
the ellipsoidal truth on a coarse lattice and keeps the planar figure when the
departure stays within ``PLANAR_AREA_TOLERANCE`` (UTM, national grids and
equal-area projections). Geographic grids and strongly distorted projections
(Web Mercator) get ``TileWeights``: the geodesic area of each
``AREA_TILE_SIZE``-pixel tile shared equally among its pixels, which bounds the
error by the within-tile variation (under 0.3 % at 60 degrees for 30 m pixels).
"""

import math
from typing import Tuple

import numpy as np
from pyproj import Geod, Transformer
from rasterio.errors import CRSError

from component.config.config import AREA_TILE_SIZE, PLANAR_AREA_TOLERANCE
from component.scripts.raster_source import NotThematicError

_GEOD = Geod(ellps="WGS84")
_PROBE_CELLS = 8  # cells per axis of the tolerance probe


def planar_pixel_area(transform, crs) -> float:
    """Pixel area in m² from the affine transform and the CRS linear unit."""
    try:
        _, factor = crs.linear_units_factor
    except CRSError:  # rasterio raises CRSError for a unit it cannot name
        factor = 1.0
    return abs(transform.a * transform.e) * factor * factor


def tile_geodesic_areas(transform, crs, row_edges, col_edges) -> np.ndarray:
    """Geodesic area (m²) of every cell of a pixel lattice.

    ``row_edges`` / ``col_edges`` are pixel offsets bounding the cells; the
    result has shape ``(len(row_edges) - 1, len(col_edges) - 1)``. Cell
    corners are mapped through the transform, then to lon/lat, and each cell
    is measured as a geodesic quadrilateral.
    """
    row_edges = np.asarray(row_edges, dtype=float)
    col_edges = np.asarray(col_edges, dtype=float)
    cols, rows = np.meshgrid(col_edges, row_edges)
    xs = transform.c + transform.a * cols + transform.b * rows
    ys = transform.f + transform.d * cols + transform.e * rows
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)

    areas = np.empty((len(row_edges) - 1, len(col_edges) - 1))
    for i in range(areas.shape[0]):
        for j in range(areas.shape[1]):
            area, _ = _GEOD.polygon_area_perimeter(
                [lon[i, j], lon[i, j + 1], lon[i + 1, j + 1], lon[i + 1, j]],
                [lat[i, j], lat[i, j + 1], lat[i + 1, j + 1], lat[i + 1, j]],
            )
            areas[i, j] = abs(area)
    return areas


def _probe_edges(size: int) -> np.ndarray:
    return np.unique(np.linspace(0, size, _PROBE_CELLS + 1).round().astype(int))


def choose_area_method(transform, crs, width: int, height: int) -> Tuple[str, float]:
    """``("planar" | "geodesic", departure)`` for a raster grid.

    ``departure`` is ``max |geodesic / planar - 1|`` over an 8 x 8 probe of
    the extent (``inf`` for a geographic CRS, whose planar figure is not an
    area at all). Raises ``NotThematicError("no_crs")`` when there is no CRS.
    """
    if crs is None:
        raise NotThematicError("no_crs")
    if crs.is_geographic:
        return "geodesic", math.inf
    rows = _probe_edges(height)
    cols = _probe_edges(width)
    geodesic = tile_geodesic_areas(transform, crs, rows, cols)
    planar = np.outer(np.diff(rows), np.diff(cols)) * planar_pixel_area(transform, crs)
    departure = float(np.max(np.abs(geodesic / planar - 1.0)))
    method = "planar" if departure <= PLANAR_AREA_TOLERANCE else "geodesic"
    return method, departure


def _tile_edges(size: int, tile: int) -> np.ndarray:
    return np.append(np.arange(0, size, tile), size)


class TileWeights:
    """Per-pixel ground area (m²) as a lattice of ``tile`` x ``tile`` blocks."""

    def __init__(
        self, transform, crs, width: int, height: int, tile: int = AREA_TILE_SIZE
    ):
        self.tile = int(tile)
        rows = _tile_edges(height, self.tile)
        cols = _tile_edges(width, self.tile)
        areas = tile_geodesic_areas(transform, crs, rows, cols)
        self.per_pixel = areas / np.outer(np.diff(rows), np.diff(cols))

    def for_window(self, window) -> np.ndarray:
        """Weights for a rasterio ``Window`` (integer-valued block window).

        Shape ``(window.height, window.width)``. Window parameters are rounded to
        absorb float noise from coordinate transformations.
        """
        row_off = round(window.row_off)
        col_off = round(window.col_off)
        height = round(window.height)
        width = round(window.width)
        rows = np.arange(row_off, row_off + height) // self.tile
        cols = np.arange(col_off, col_off + width) // self.tile
        return self.per_pixel[np.ix_(rows, cols)]
