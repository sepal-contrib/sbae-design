import logging

import ipyleaflet
import solara
from localtileserver import TileClient, get_leaflet_tile_layer
from sepal_ui.mapping import SepalMap
from sepal_ui.sepalwidgets.vue_app import ThemeToggle

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


def _tile_client_kwargs_for_runtime(is_voila: bool | None = None) -> dict:
    """Return TileClient browser URL options for the current runtime."""
    if is_voila is None:
        is_voila = solara.util.is_running_in_voila()

    if not is_voila:
        return {}

    return {
        "client_host": "127.0.0.1",
        "client_port": True,
        "cors_all": True,
    }


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

        client = TileClient(path, **_tile_client_kwargs_for_runtime())
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

    def add_sample_points(self, points_data):
        """Add sample points layer."""
        if self.sample_points_layer:
            logger.debug("Removing existing sample points layer.")
            self.remove_layer(self.sample_points_layer)

        if not points_data.empty:
            markers = []
            logger.debug(f"Adding {len(points_data)} sample points to the map.")
            for _, point in points_data.iterrows():
                marker = ipyleaflet.Marker(
                    location=(point["latitude"], point["longitude"]),
                    title=f"Class: {point.get('map_code', 'Unknown')}",
                )
                markers.append(marker)

            logger.debug("Creating marker cluster for sample points.")

            self.sample_points_layer = ipyleaflet.MarkerCluster(markers=markers)
            self.add_layer(self.sample_points_layer)
