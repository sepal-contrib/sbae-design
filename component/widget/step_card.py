"""Vue-based Step Card Component.

A card component for displaying workflow steps with large icons using Vue template.
"""

from typing import Callable, Optional

import solara


@solara.component_vue("vue/StepCard.vue")
def StepCard(
    number: str,
    title: str,
    icon: str,
    elevation: int = 3,
    height: str = "280px",
    event_click: Optional[Callable[[int], None]] = None,
):
    """A Vue-based card component for displaying workflow steps with large icons.

    Args:
        number: Step number (e.g., "1")
        title: Step title (e.g., "Upload Map")
        icon: Material Design Icon name (e.g., "mdi-upload")
        elevation: Card elevation (shadow depth)
        height: Card height in CSS units
        event_click: Click event handler function
    """
    pass
