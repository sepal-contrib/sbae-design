import solara

from component.model import app_state
from component.widget.custom_widgets import DownloadMenu


@solara.component
def Export():
    """Export options component - self-contained with its own logic."""
    points = app_state.sample_points.value
    if points is None or points.empty:
        solara.Info("Generate sample points first to enable export.")
        return

    items = [
        (
            "Sample points (CSV)",
            app_state.export_csv(),
            "sample_points.csv",
            "text/csv",
        ),
        (
            "Sample points (GeoJSON)",
            app_state.export_geojson(),
            "sample_points.geojson",
            "application/geo+json",
        ),
    ]
    DownloadMenu(items)
