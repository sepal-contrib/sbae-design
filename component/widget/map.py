import logging

import ipyleaflet
from sepal_ui.mapping import SepalMap
from sepal_ui.sepalwidgets.vue_app import ThemeToggle

logger = logging.getLogger("sbae.map")


class SbaeMap(SepalMap):
    """SBAE Map class extending SepalMap for map visualization and interactions."""

    def __init__(self, theme_toggle: ThemeToggle, gee: bool = False):
        super().__init__(fullscreen=True, theme_toggle=theme_toggle, gee=gee)

        self.classification_layer = None
        self.sample_points_layer = None

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
