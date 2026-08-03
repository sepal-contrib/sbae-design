"""SBAE (Sampling-Based Area Estimation) Map Application.

A Solara web application using the MapApp layout with a map background,
left drawer for workflow steps, and right panel for tools and exports.
"""

# ruff: noqa: E402
# PROJ_DATA must be set before importing rasterio/pyproj to avoid CRS errors
# when system PROJ_DATA points to an incompatible proj.db (e.g., in SEPAL)
import os
import sys

env_prefix = sys.prefix
proj_data = os.path.join(env_prefix, "share", "proj")
if os.path.exists(proj_data):
    os.environ["PROJ_DATA"] = proj_data

import logging

import solara
from sepal_ui.logger import setup_logging
from sepal_ui.sepalwidgets.vue_app import MapApp, ThemeToggle
from sepal_ui.solara import (
    ThemeState,
    TileBridge,
    setup_sessions,
    setup_solara_server,
    setup_theme_colors,
)
from sepal_ui.solara.notifications import NotificationProvider
from solara.lab.components.theming import theme

from component.model.app_model import AppModel
from component.tile.upload import RasterMapWatcher
from component.widget.map import PointsLegend, SbaeMap
from component.widget.notification_bridge import ErrorToastBridge
from component.widget.sample_configuration import SampleConfiguration

logger = setup_logging(logger_name="sbae")

# set debug level
logger.setLevel(logging.DEBUG)
logger.debug("SBAE Map App initialized")
logger.debug("Solara version: %s", solara.__version__)

setup_solara_server()

# GEE Configuration - Set to True to enable Google Earth Engine features
USE_GEE = False


@solara.lab.on_kernel_start
def on_kernel_start():
    return setup_sessions()


@solara.component
# @with_sepal_sessions(module_name="sbae_app")
def Page():
    """Main SBAE application page using MapApp layout."""
    # pysepal's MapApp requires a per-kernel ThemeState. In a local (non-SEPAL)
    # run the session manager is active but has no theme_state component, so
    # get_current_theme_state() would raise; provide an explicit one instead.
    theme_state = solara.use_memo(ThemeState, [])

    # Notification system (pysepal): mount the provider once at the app root,
    # before any component that calls use_notifications(). Kept in the same page
    # as MapApp so the task pill can track the right-panel offset. It takes the
    # same ThemeState as MapApp: without it the provider falls back to a
    # process-local default and the toasts/pill stay light under a dark app.
    NotificationProvider(theme_state=theme_state)
    ErrorToastBridge()
    TileBridge()

    app_model = AppModel()

    setup_theme_colors()
    theme_toggle = ThemeToggle()
    theme_toggle.observe(lambda e: setattr(theme, "dark", e["new"]), "dark")
    sbae_map = SbaeMap(theme_toggle=theme_toggle, gee=USE_GEE)

    RasterMapWatcher(sbae_map)
    # Floating legend overlay for the sample/reference points (bottom-center).
    PointsLegend()

    steps_data = [
        {
            "id": 4,
            "name": "Sample design",
            "icon": "mdi-tune",
            "display": "step",
            "content": [],
            "right_panel_action": "toggle",
        },
    ]

    # Right panel configuration
    right_panel_config = {
        "title": "Sample design tools",
        "icon": "mdi-tools",
        "width": 450,
        "toggle_icon": "mdi-chevron-left",
        "is_open": True,
    }

    # Right panel content: a single section holding the tabbed Sample
    # Configuration (Design | Analysis). The design-phase outputs (summary /
    # generate points / export) now live inside the Design tab, so they no
    # longer appear while the user is on the Analysis tab.
    right_panel_content = [
        {
            "content": [SampleConfiguration(sbae_map, theme_toggle=theme_toggle)],
        },
    ]

    # Create the MapApp with the shared map instance
    MapApp.element(
        app_title="SBAE - Sampling-Based Area Estimation",
        app_icon="mdi-map-marker-radius",
        main_map=[sbae_map],
        steps_data=steps_data,
        initial_step=4,
        theme_toggle=[theme_toggle],
        theme_state=theme_state,
        dialog_width=900,
        right_panel_config=right_panel_config,
        right_panel_content=right_panel_content,
        right_panel_open=True,
        is_pinned=False,
        repo_url="https://github.com/your-repo/sbae-tool",
        docs_url="https://your-docs-url.com/sbae",
        model=app_model,
    )


# Routes for the application
routes = [
    solara.Route(path="/", component=Page, label="SBAE Tool"),
]
