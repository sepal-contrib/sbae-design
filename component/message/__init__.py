"""The app's message catalog, resolved for the connection's locale.

Components read messages through :func:`use_translator`, so a language change
re-renders them in place. Helpers that are not components take an optional
``ms`` and fall back to :func:`get_translator`, which resolves English.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

import solara
from pysepal.solara import use_locale
from pysepal.translator import Translator

MESSAGE_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def get_translator(locale: str = "en") -> Translator:
    """Return the catalog for ``locale``; untranslated keys fall back to English."""
    return Translator(MESSAGE_DIR, target=locale)


def use_translator() -> Translator:
    """Reactively return the catalog for the locale resolved in the browser."""
    locale = use_locale()
    return solara.use_memo(lambda: get_translator(locale), [locale])


def available_locales() -> List[str]:
    """Locale codes the app ships a catalog for, for ``MapApp(locales=...)``."""
    return get_translator().available_locales()
