import logging
import tempfile

from localtileserver import TileClient, get_leaflet_tile_layer
from sepal_ui.mapping import SepalMap
from sepal_ui.sepalwidgets.vue_app import ThemeToggle

from component.scripts.vector_tiles import (
    VectorTileError,
    build_points_pmtiles_layer,
)

logger = logging.getLogger("sbae.map")


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#rrggbb' to an (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _build_class_colormap(class_colors: dict) -> dict:
    """Build a discrete {pixel_value: (r, g, b, a)} LUT for a categorical raster.

    Every class present in ``class_colors`` is rendered opaque, including code 0
    and codes above 255 -- area calculation treats those as valid, sampleable
    classes, so they must be visible on the map. Values not in ``class_colors``
    render transparent (background).
    """
    colormap = {i: (0, 0, 0, 0) for i in range(256)}
    for code, hex_color in class_colors.items():
        colormap[int(code)] = (*_hex_to_rgb(hex_color), 255)
    return colormap


class SbaeMap(SepalMap):
    """SBAE Map class extending SepalMap for map visualization and interactions."""

    def __init__(self, theme_toggle: ThemeToggle, gee: bool = False):
        super().__init__(fullscreen=True, theme_toggle=theme_toggle, gee=gee)

        self.classification_layer = None
        self.sample_points_layer = None

    def add_class_raster(
        self,
        path,
        class_colors,
        layer_name: str = "Classification Map",
        key: str = "clas",
        opacity: float = 1.0,
        fit_bounds: bool = True,
    ):
        """Add a categorical raster with exact per-class colors.

        Unlike ``add_raster`` (which applies a continuous inferno colormap
        stretched across the value range and renders sparse/low-value class
        maps as black), this builds a discrete lookup table so each class
        value gets its own color. Any value not in ``class_colors`` is rendered
        transparent (background); classes 0 and > 255 are colored like any other.

        Args:
            path: path to the (optimized) raster file.
            class_colors: mapping of class code -> '#rrggbb' hex color.
            layer_name: display name of the layer.
            key: unequivocal key of the layer (for later removal).
            opacity: layer opacity, default 1.0.
            fit_bounds: whether to recenter/zoom onto the raster.
        """
        if not class_colors:
            logger.warning(
                "add_class_raster called without class_colors; "
                "falling back to add_raster for %s",
                path,
            )
            return self.add_raster(
                path,
                layer_name=layer_name,
                key=key,
                opacity=opacity,
                fit_bounds=fit_bounds,
            )

        # Discrete LUT: transparent everywhere unless it's a known class. Every
        # class in class_colors is colored, including code 0 and codes > 255.
        colormap = _build_class_colormap(class_colors)

        # localtileserver won't accept a raw {value: rgba} dict as `colormap`;
        # it must be registered server-side first, which yields a "custom:<hash>"
        # key. This is the only path that preserves per-class alpha (the
        # matplotlib-Colormap path forces alpha=1, losing transparency).
        try:
            from localtileserver.tiler.palettes import register_colormap

            colormap_arg = register_colormap(colormap)
        except Exception:
            logger.warning(
                "localtileserver register_colormap unavailable; falling back "
                "to add_raster (classes may render dark) for %s",
                path,
            )
            return self.add_raster(
                path,
                layer_name=layer_name,
                key=key,
                opacity=opacity,
                fit_bounds=fit_bounds,
            )

        client = TileClient(path)
        layer = get_leaflet_tile_layer(
            client,
            colormap=colormap_arg,
            name=layer_name,
            opacity=opacity,
            max_zoom=20,
        )
        self.add_layer(layer, key=key)
        self.classification_layer = layer
        layer.raster = str(path)

        if fit_bounds:
            self.center = client.center()
            self.zoom = client.default_zoom

        return layer

    def build_sample_points_layer(self, points_data, class_colors=None):
        """Build (off the UI thread) a PMTiles layer for the sample points.

        Returns ``None`` for an empty DataFrame. Does NOT mutate the map, so it
        is safe to call from a worker thread. Raises ``VectorTileError`` on
        failure. ``class_colors=None`` resolves to ``app_state.class_colors``.
        """
        if points_data is None or points_data.empty:
            return None
        if class_colors is None:
            from component.model import app_state

            class_colors = app_state.class_colors.value or {}
        dest_dir = tempfile.mkdtemp(prefix="sbae_points_")
        return build_points_pmtiles_layer(points_data, class_colors, dest_dir=dest_dir)

    def attach_sample_points_layer(self, layer):
        """Swap the sample-points layer on the map (call on the main thread)."""
        if self.sample_points_layer is not None:
            self.remove_layer(self.sample_points_layer)
            self.sample_points_layer = None
        if layer is not None:
            self.add_layer(layer, key="sample_pts")
            self.sample_points_layer = layer

    def add_sample_points(self, points_data, class_colors=None):
        """Build + attach the sample-points PMTiles layer.

        For callers already off the UI thread (e.g. the analysis derivation).
        On failure, notify and skip -- the map shows no points layer; other
        results are unaffected.
        """
        from component.model import app_state

        try:
            layer = SbaeMap.build_sample_points_layer(self, points_data, class_colors)
        except VectorTileError as e:
            logger.warning("Sample points layer failed: %s", e)
            app_state.add_error(f"Could not render sample points on the map: {e}")
            return
        SbaeMap.attach_sample_points_layer(self, layer)
