import solara

from component.message import use_translator
from component.widget.step_card import StepCard


@solara.component
def LandingTile(app_model):
    """Landing dialog with workflow steps as cards."""
    ms = use_translator()

    def handle_step_click(step_number: int):
        """Handle click on a step card to navigate to that step."""
        app_model.current_step = step_number

    with solara.Column():
        with solara.Column(style={"text-align": "center", "margin-bottom": "30px"}):
            solara.HTML(tag="h1", unsafe_innerHTML=ms.landing.title)

        # Create workflow step cards
        workflow_steps = [
            {
                "number": "1",
                "title": ms.landing.step_design,
                "icon": "mdi-pencil",
                "step_id": 4,
            },
            {
                "number": "2",
                "title": ms.landing.step_analyze,
                "icon": "mdi-chart-bar",
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
