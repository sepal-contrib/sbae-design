"""Tests for the error->toast notification bridge."""

import solara
from pysepal.solara import NotificationProvider
from pysepal.solara.notifications.bus import get_current_bus

from component.model import app_state
from component.widget.notification_bridge import ErrorToastBridge


@solara.component
def _App():
    NotificationProvider()
    ErrorToastBridge()


def _toast_messages():
    return [toast.message for toast in get_current_bus().toasts.value]


def test_error_toast_bridge_renders_under_the_provider():
    app_state.error_messages.value = []

    _, rc = solara.render(_App(), handle_error=False)

    assert _toast_messages() == []
    rc.close()


def test_error_toast_bridge_toasts_each_new_error():
    app_state.error_messages.value = []
    _, rc = solara.render(_App(), handle_error=False)

    app_state.add_error("boom happened")

    assert "boom happened" in _toast_messages()
    rc.close()
