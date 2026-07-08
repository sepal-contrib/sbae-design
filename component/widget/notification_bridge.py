"""Bridge app_state error messages to the pysepal notification system."""

import logging

import solara
from sepal_ui.solara.notifications import use_notifications

from component.model import app_state

logger = logging.getLogger("sbae.notifications")


@solara.component
def ErrorToastBridge():
    """Surface ``app_state.error_messages`` as notification toasts.

    ``app_state.add_error()`` is called from many places across the app, but the
    list was never rendered anywhere, so errors were silently dropped. This
    component watches the list and emits a toast (and a log line) for each new
    entry. Without a mounted ``NotificationProvider`` the hook is a no-op, so it
    is safe to render in isolation.
    """
    notifications = use_notifications()
    seen = solara.use_ref(0)
    errors = app_state.error_messages.value

    def emit():
        n = len(errors)
        if n < seen.current:  # list was cleared/reset
            seen.current = 0
        for message in errors[seen.current :]:
            logger.warning("app error surfaced: %s", message)
            notifications.error(message)
        seen.current = n

    solara.use_effect(emit, [errors])
