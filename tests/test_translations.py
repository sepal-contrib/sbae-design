"""The message catalog and the locale wiring behind it.

pysepal 4 resolves the locale in the browser and pushes it into a scope-keyed
``LocaleState``; the app rebuilds its ``Translator`` from that. These guard the
two halves: every shipped catalog must load, and a language change must reach a
mounted component.
"""

import ipyvuetify as v
import pytest
import solara
from box import BoxKeyError
from pysepal.solara import get_current_locale_state

from component.message import MESSAGE_DIR, available_locales, get_translator
from component.widget.aoi_upload_selector import UploadDialogCard


@pytest.fixture(autouse=True)
def _reset_locale():
    """Keep a locale switch from leaking into the rest of the suite."""
    state = get_current_locale_state()
    yield
    state.set_locale("en")


def _button_labels(rc) -> str:
    return " ".join(str(c) for b in rc.find(v.Btn).widgets for c in (b.children or []))


def test_every_shipped_locale_loads():
    """A catalog folder the picker offers must build a usable Translator.

    ``available_locales`` feeds ``MapApp(locales=...)``, so a folder that fails
    to load would be selectable and then blank the UI.
    """
    for locale in available_locales():
        assert get_translator(locale).app.title


def test_available_locales_are_the_catalog_folders_only():
    """``__pycache__`` sits beside the catalogs and must never be offered."""
    assert set(available_locales()) == {"en", "es", "fr"}
    assert (MESSAGE_DIR / "en" / "locale.json").exists()


def test_untranslated_key_falls_back_to_english():
    """The es/fr catalogs are partial on purpose; missing keys use English."""
    assert (
        get_translator("es").analysis.calculate == get_translator().analysis.calculate
    )


def test_browser_locale_variant_resolves_to_a_shipped_catalog():
    """``navigator.language`` reports es-CL where the app only ships ``es``."""
    assert get_translator("es-CL").common.close == "Cerrar"


def test_unknown_locale_resolves_to_english():
    assert get_translator("de").common.close == "Close"


def test_missing_key_raises_instead_of_rendering_blank():
    """A typo'd key must fail loudly rather than render an empty string."""
    with pytest.raises(BoxKeyError):
        get_translator().upload.no_such_key


def test_language_change_re_renders_a_mounted_component():
    """Switching locale re-renders in place -- no reload, no remount.

    ``use_translator`` builds the catalog from ``use_locale``, so the component
    must pick up a ``LocaleState`` change pushed from the language selector.
    """
    _, rc = solara.render(
        UploadDialogCard(sbae_map=None, on_close=lambda: None), handle_error=False
    )
    assert "Close" in _button_labels(rc)

    get_current_locale_state().set_locale("es")

    assert "Cerrar" in _button_labels(rc)
