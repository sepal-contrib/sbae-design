import ipyleaflet
from sepal_ui.mapping import SepalMap
from sepal_ui.sepalwidgets.vue_app import ThemeToggle


class SbaeMap:
    """SBAE Map class to handle map visualization and interactions."""

    def __init__(self, theme_toggle: ThemeToggle):
        # Create the SepalMap with fullscreen enabled
        self.map = SepalMap(fullscreen=True, theme_toggle=theme_toggle)

        # Layers for different data types
        self.classification_layer = None
        self.sample_points_layer = None

    def add_classification_layer(self, layer_data):
        """Add classification map layer."""
        if self.classification_layer:
            self.map.remove_layer(self.classification_layer)

        # TODO: Implement actual layer creation from uploaded data
        self.classification_layer = layer_data
        if self.classification_layer:
            self.map.add_layer(self.classification_layer)

    def add_sample_points(self, points_data):
        """Add sample points layer."""
        if self.sample_points_layer:
            self.map.remove_layer(self.sample_points_layer)

        if not points_data.empty:
            # Create markers for sample points
            markers = []
            for _, point in points_data.iterrows():
                marker = ipyleaflet.Marker(
                    location=(point["latitude"], point["longitude"]),
                    title=f"Class: {point.get('map_code', 'Unknown')}",
                )
                markers.append(marker)

            # Create marker cluster
            self.sample_points_layer = ipyleaflet.MarkerCluster(markers=markers)
            self.map.add_layer(self.sample_points_layer)

    def fit_bounds(self, bounds):
        """Fit map to given bounds."""
        if bounds:
            self.map.fit_bounds(bounds)

    def get_map_widget(self):
        """Get the map widget for embedding in UI."""
        return self.map
