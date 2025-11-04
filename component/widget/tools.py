import solara

from component.model import app_state
from component.widget.custom_widgets import error_display


@solara.component
def Tools():
    """Right panel tools and utilities."""
    with solara.Column():
        # Error display
        if app_state.error_messages.value:
            with solara.Card("⚠️ Errors"):
                for error in app_state.error_messages.value:
                    error_display(error, "error")

                solara.Button(
                    "Clear Errors",
                    on_click=app_state.clear_errors,
                    color="secondary",
                    outlined=True,
                )

        # Processing status
        if app_state.processing_status.value:
            with solara.Card("📊 Status"):
                solara.Markdown(f"**Current:** {app_state.processing_status.value}")
