"""Tests for the error->toast notification bridge."""

import solara

from component.model import app_state
from component.widget.notification_bridge import ErrorToastBridge


def test_error_toast_bridge_renders_without_provider():
    # No NotificationProvider mounted -> use_notifications() returns a no-op;
    # the bridge must still render without raising.
    app_state.error_messages.value = []

    _, rc = solara.render(ErrorToastBridge(), handle_error=False)

    assert rc is not None


def test_error_toast_bridge_survives_new_errors():
    app_state.error_messages.value = []

    solara.render(ErrorToastBridge(), handle_error=False)
    # Adding an error triggers the bridge effect; with no provider it is a no-op
    # and must not crash.
    app_state.add_error("boom happened")

    assert "boom happened" in app_state.error_messages.value
