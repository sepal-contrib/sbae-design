"""SBAE UI Components Module.

Contains reusable Solara components for the SBAE application.
"""

import solara
from solara.alias import rv


def error_display(error_message: str, error_type: str = "error") -> None:
    """Display error messages with appropriate styling.

    Args:
        error_message: Error message to display
        error_type: Type of error (error, warning, info)
    """
    with rv.Alert(type=error_type, text=True):
        solara.Markdown(f"**{error_type.title()}:** {error_message}")


def info_panel(title: str, content: str, collapsible: bool = True) -> None:
    """Display informational content in an expandable panel.

    Args:
        title: Panel title
        content: Panel content (markdown)
        collapsible: Whether panel can be collapsed
    """
    if collapsible:
        with solara.Details(title):
            solara.Markdown(content)
    else:
        with solara.Card(title):
            solara.Markdown(content)
