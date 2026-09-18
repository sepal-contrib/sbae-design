import solara

from component.message import msg
from component.model import app_state
from component.widget.custom_widgets import DownloadMenu


@solara.component
def Export():
    """Export options component - self-contained with its own logic."""
    points = app_state.sample_points.value
    if points is None or points.empty:
        solara.Info(msg("design.export_files.empty"))
        return

    items = [
        (
            msg("design.export_files.csv"),
            app_state.export_csv(),
            "sample_points.csv",
            "text/csv",
        ),
        (
            msg("design.export_files.geojson"),
            app_state.export_geojson(),
            "sample_points.geojson",
            "application/geo+json",
        ),
    ]
    DownloadMenu(items)
