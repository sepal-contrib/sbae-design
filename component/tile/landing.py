import solara

from component.widget.step_card import StepCard


@solara.component
def LandingTile(app_model):
    """Landing dialog with workflow steps as cards."""

    def handle_step_click(step_number: int):
        """Handle click on a step card to navigate to that step."""
        app_model.current_step = step_number

    with solara.Column():
        # Centered title
        with solara.Column(style={"text-align": "center", "margin-bottom": "30px"}):
            solara.Markdown("# 🌍 SBAE - Sampling-Based Area Estimation Design")

        # Create workflow step cards
        workflow_steps = [
            {"number": "2", "title": "Upload Map", "icon": "mdi-upload", "step_id": 4},
            {
                "number": "3",
                "title": "Export Results",
                "icon": "mdi-download",
                "step_id": 3,
            },
        ]

        # Display cards in horizontal layout with taller cards
        with solara.Columns([4, 4, 4], gutters=True):
            for step in workflow_steps:
                StepCard(
                    number=step["number"],
                    title=step["title"],
                    icon=step["icon"],
                    elevation=3,
                    height="200px",
                    event_click=lambda step_id=step["step_id"]: handle_step_click(
                        step_id
                    ),
                )
