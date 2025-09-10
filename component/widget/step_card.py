"""Vue Step Card Component.

A wrapper for the StepCard.vue component to use in Solara applications.
"""

from pathlib import Path

import solara


@solara.component_vue(str(Path(__file__).parent / "vue" / "StepCard.vue"), vuetify=True)
def StepCard(
    number: str, title: str, icon: str, elevation: int = 3, height: str = "280px"
):
    """A card component for displaying workflow steps with large icons.

    Args:
        number: Step number (e.g., "1")
        title: Step title (e.g., "Upload Map")
        icon: Material Design Icon name (e.g., "mdi-upload")
        elevation: Card elevation (shadow depth)
        height: Card height in CSS units
    """
    pass
