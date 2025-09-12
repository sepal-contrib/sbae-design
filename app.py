"""SBAE (Sampling-Based Area Estimation) Map Application.

A Solara web application using the MapApp layout with a map background,
left drawer for workflow steps, and right panel for tools and exports.
"""

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
from component.tile.class_editor import ClassEditorTile
from component.tile.export import ExportTile
from component.tile.landing import LandingTile
from component.tile.point_generator import PointGenerationTile
from component.tile.sample_calculation import SampleCalculationTile
from component.tile.upload import UploadTile
from component.widget.map import SbaeMap
from component.widget.summary import Summary
from component.widget.tools import Tools

logger = setup_logging(logger_name="sbae")
logger.debug("SBAE Map App initialized")


setup_solara_server()


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
    sbae_map = SbaeMap(theme_toggle=theme_toggle)

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
            "id": 2,
            "name": "1. Upload Map",
            "icon": "mdi-upload",
            "display": "dialog",
            "content": UploadTile(sbae_map),
            "width": 900,
            "actions": [
                {
                    "label": "Back",
                    "next": 1,
                    "cancel": True,
                },
                {"label": "Next", "next": 3},
            ],
        },
        {
            "id": 3,
            "name": "2. Edit Classes",
            "icon": "mdi-pencil",
            "display": "dialog",
            "content": ClassEditorTile(),
            "width": 900,
            "actions": [
                {
                    "label": "Back",
                    "next": 1,
                    "cancel": True,
                },
                {"label": "Next", "next": 4},
            ],
        },
        {
            "id": 4,
            "name": "3. Calculate Samples",
            "icon": "mdi-calculator",
            "display": "dialog",
            "content": SampleCalculationTile(),
            "width": 900,
            "actions": [
                {
                    "label": "Back",
                    "next": 1,
                    "cancel": True,
                },
                {"label": "Next", "next": 5},
            ],
        },
        {
            "id": 5,
            "name": "4. Generate Points",
            "icon": "mdi-map-marker-multiple",
            "display": "dialog",
            "content": PointGenerationTile(sbae_map),
            "width": 900,
            "actions": [
                {
                    "label": "Back",
                    "next": 1,
                    "cancel": True,
                },
                {"label": "Next", "next": 6},
            ],
        },
        {
            "id": 6,
            "name": "5. Export Results",
            "icon": "mdi-download",
            "display": "dialog",
            "content": ExportTile(),
            "width": 900,
            "actions": [
                {
                    "label": "Back",
                    "next": 1,
                    "cancel": True,
                },
                {"label": "Finish", "close": True},
            ],
        },
        {
            "id": 7,
            "name": "Summary",
            "icon": "mdi-database",
            "display": "step",
            "content": [],
            "right_panel_action": "toggle",
        },
    ]

    # Right panel configuration
    right_panel_config = {
        "title": "SBAE Tools",
        "icon": "mdi-tools",
        "width": 450,
        "description": "Progress tracking, statistics, and application tools for the SBAE workflow.",
        "toggle_icon": "mdi-chevron-left",
    }

    # Right panel content sections
    right_panel_content = [
        {
            "title": "Progress & Summary",
            "icon": "mdi-progress-check",
            "content": [Summary()],
            "description": "Track your progress through the SBAE workflow and view summary statistics.",
        },
        {
            "title": "Tools & Settings",
            "icon": "mdi-cog",
            "content": [Tools()],
            "divider": True,
            "description": "Application tools, error management, and reset functionality.",
        },
    ]

    # Create the MapApp with the shared map instance
    MapApp.element(
        app_title="SBAE - Sampling-Based Area Estimation",
        app_icon="mdi-map-marker-radius",
        main_map=[sbae_map.get_map_widget()],  # Pass SepalMap widget directly
        steps_data=steps_data,
        initial_step=1,  # Start with About dialog
        theme_toggle=[theme_toggle],
        dialog_width=900,
        right_panel_config=right_panel_config,
        right_panel_content=right_panel_content,
        repo_url="https://github.com/your-repo/sbae-tool",
        docs_url="https://your-docs-url.com/sbae",
        model=app_model,
    )


# Routes for the application
routes = [
    solara.Route(path="/", component=Page, label="SBAE Tool"),
]
