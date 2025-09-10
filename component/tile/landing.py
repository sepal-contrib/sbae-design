import solara

from component.widget.step_card import StepCard


@solara.component
def LandingTile(current_dialog, set_current):
    """Landing dialog with workflow steps as cards."""
    with solara.Column():
        # Centered title
        with solara.Column(style={"text-align": "center", "margin-bottom": "30px"}):
            solara.Markdown("# 🌍 SBAE - Sampling-Based Area Estimation")

        # Create workflow step cards
        workflow_steps = [
            {"number": "1", "title": "Upload Map", "icon": "mdi-upload"},
            {"number": "2", "title": "Edit Classes", "icon": "mdi-pencil"},
            {"number": "3", "title": "Calculate Samples", "icon": "mdi-calculator"},
            {
                "number": "4",
                "title": "Generate Points",
                "icon": "mdi-map-marker-multiple",
            },
            {"number": "5", "title": "Export Results", "icon": "mdi-download"},
        ]

        # Display cards in horizontal layout with taller cards
        with solara.Columns([2, 2, 2, 2, 2], gutters=True):
            for step in workflow_steps:
                StepCard(
                    number=step["number"],
                    title=step["title"],
                    icon=step["icon"],
                    elevation=3,
                    height="280px",
                )
