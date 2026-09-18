"""The app's message catalog.

``msg("dotted.key", **values)`` renders a message in the connection's locale.
Called during a render it subscribes the component, so a language change
re-renders it in place.
"""

from pathlib import Path

from pysepal.i18n import catalog

MESSAGE_DIR = Path(__file__).parent

messages = catalog(MESSAGE_DIR)
msg = messages.msg
