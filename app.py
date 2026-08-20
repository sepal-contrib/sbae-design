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

# Both tile servers bind 127.0.0.1 inside the kernel, so the browser needs a route
# that reaches them. SEPAL sets LOCALTILESERVER_CLIENT_PREFIX to a
# jupyter-server-proxy route that forwards any port in the sandbox, so it carries
# PMTiles as well as raster tiles -- but vectortileserver never autodetects one,
# and left alone its layers keep a URL the browser cannot reach, so the points
# silently never arrive. Only borrow the generic /proxy/{port} form: it forwards
# any port by construction, while a route namespaced to one server (localtileserver's
# own autodetected prefix) would not serve a vector port.
_raster_prefix = os.environ.get("LOCALTILESERVER_CLIENT_PREFIX")
if _raster_prefix and "/proxy/{port}" in _raster_prefix:
    os.environ.setdefault("VECTORTILESERVER_CLIENT_PREFIX", _raster_prefix)

import logging

import solara
from pysepal.logger import setup_logging
from pysepal.sepalwidgets.vue_app import MapApp
from pysepal.solara import (
    NotificationProvider,
    get_current_theme_state,
    setup_sessions,
    setup_solara_server,
    setup_theme_colors,
)

from component.message import available_locales, get_translator, use_translator
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
    theme_state = get_current_theme_state()
    ms = use_translator()

    # Notification system (pysepal): mount the provider once at the app root,
    # before any component that calls use_notifications(). Kept in the same page
    # as MapApp so the task pill can track the right-panel offset. It takes the
    # same ThemeState as MapApp: without it the provider falls back to a
    # process-local default and the toasts/pill stay light under a dark app.
    NotificationProvider(theme_state=theme_state)
    ErrorToastBridge()

    app_model = AppModel()

    setup_theme_colors()
    sbae_map = SbaeMap(theme_state=theme_state, gee=USE_GEE)

    RasterMapWatcher(sbae_map)
    # Floating legend overlay for the sample/reference points (bottom-center).
    PointsLegend()

    steps_data = [
        {
            "id": 4,
            "name": ms.app.step_sample_design,
            "icon": "mdi-tune",
            "display": "step",
            "content": [],
            "right_panel_action": "toggle",
        },
    ]

    # Right panel configuration
    right_panel_config = {
        "title": ms.app.right_panel_title,
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
            "content": [SampleConfiguration(sbae_map, theme_state=theme_state)],
        },
    ]

    # Create the MapApp with the shared map instance
    MapApp.element(
        app_title=ms.app.title,
        app_icon="mdi-map-marker-radius",
        locales=available_locales(),
        main_map=[sbae_map],
        steps_data=steps_data,
        initial_step=4,
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


# Routes for the application. The label is read once at import, outside any
# render, so it stays in the default locale.
routes = [
    solara.Route(path="/", component=Page, label=get_translator().app.route_label),
]
