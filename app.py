"""SBAE (Sampling-Based Area Estimation) Map Application.

A Solara web application using the MapApp layout with a map background,
left drawer for workflow steps, and right panel for tools and exports.
"""

import logging

import solara
from sepal_ui.logger import setup_logging
from sepal_ui.sepalwidgets.vue_app import MapApp, ThemeToggle
from sepal_ui.solara import (
    setup_sessions,
    setup_solara_server,
    setup_theme_colors,
)
from solara.lab.components.theming import theme

from component.model.app_model import AppModel
from component.tile.export import Export
from component.tile.landing import LandingTile
from component.widget.map import SbaeMap
from component.widget.point_generation import PointGeneration
from component.widget.sample_configuration import SampleConfiguration
from component.widget.summary import Summary

logger = setup_logging(logger_name="sbae")

# set debug level
logger.setLevel(logging.DEBUG)
logger.debug("SBAE Map App initialized")
logger.debug("Solara version: %s", solara.__version__)

setup_solara_server()

# GEE Configuration - Set to True to enable Google Earth Engine features
USE_GEE = True


@solara.lab.on_kernel_start
def on_kernel_start():
    return setup_sessions()


@solara.component
# @with_sepal_sessions(module_name="sbae_app")
def Page():
    """Main SBAE application page using MapApp layout."""
    app_model = AppModel()

    setup_theme_colors()
    theme_toggle = ThemeToggle()
    theme_toggle.observe(lambda e: setattr(theme, "dark", e["new"]), "dark")
    sbae_map = SbaeMap(theme_toggle=theme_toggle, gee=USE_GEE)

    steps_data = [
        {
            "id": 1,
            "name": "Getting Started",
            "icon": "mdi-rocket",
            "display": "dialog",
            "content": LandingTile(app_model),
            "width": 900,
        },
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

    # Right panel content sections
    right_panel_content = [
        {
            "title": "Sample Configuration",
            "icon": "mdi-tune",
            "content": [SampleConfiguration(sbae_map)],
        },
        {
            "title": "Summary",
            "icon": "mdi-progress-check",
            "content": [Summary(theme_toggle=theme_toggle)],
        },
        {
            "title": "Generate Points",
            "icon": "mdi-map-marker-multiple",
            "content": [PointGeneration(sbae_map)],
            "description": "Generate sample points based on calculated sample sizes.",
        },
        {
            "title": "Export Results",
            "icon": "mdi-download",
            "content": [Export()],
            "description": "Generate sample points based on calculated sample sizes.",
        },
    ]

    # Create the MapApp with the shared map instance
    MapApp.element(
        app_title="SBAE - Sampling-Based Area Estimation",
        app_icon="mdi-map-marker-radius",
        main_map=[sbae_map],
        steps_data=steps_data,
        initial_step=1,
        theme_toggle=[theme_toggle],
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
