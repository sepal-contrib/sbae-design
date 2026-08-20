"""The message catalog and the locale wiring behind it.

pysepal 4 resolves the locale in the browser and pushes it into a scope-keyed
``LocaleState``; the app rebuilds its ``Translator`` from that. These guard the
two halves: every shipped catalog must load, and a language change must reach a
mounted component.
"""

import ast
import json
from pathlib import Path

import ipyvuetify as v
import pytest
import solara
from box import Box, BoxKeyError
from pysepal.solara import get_current_locale_state
from pysepal.translator import Translator

from component.message import MESSAGE_DIR, available_locales, get_translator
from component.widget.aoi_upload_selector import UploadDialogCard

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRANSLATED_LOCALES = ("es", "fr")


def _leaves(node, prefix=""):
    """Flatten a catalog to ``{dotted key: message}``."""
    flat = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_leaves(value, path))
        else:
            flat[path] = value
    return flat


def _catalog(locale):
    return _leaves(json.loads((MESSAGE_DIR / locale / "locale.json").read_text()))


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


def test_untranslated_key_falls_back_to_english(tmp_path):
    """A key a target catalog omits still renders, in English.

    Built against a synthetic catalog rather than the shipped ones so it keeps
    guarding the fallback once es/fr are complete.
    """
    (tmp_path / "en").mkdir()
    (tmp_path / "xx").mkdir()
    (tmp_path / "en" / "locale.json").write_text(
        json.dumps({"a": {"translated": "one", "untranslated": "two"}})
    )
    (tmp_path / "xx" / "locale.json").write_text(
        json.dumps({"a": {"translated": "uno"}})
    )

    ms = Translator(tmp_path, target="xx")

    assert ms.a.translated == "uno"
    assert ms.a.untranslated == "two"


@pytest.mark.parametrize("locale", _TRANSLATED_LOCALES)
def test_translated_catalog_has_the_same_keys_as_english(locale):
    """Missing keys silently fall back; orphans silently rot. Neither is visible."""
    english, translated = set(_catalog("en")), set(_catalog(locale))

    assert not english - translated, f"{locale} is missing keys"
    assert not translated - english, f"{locale} has keys English no longer defines"


@pytest.mark.parametrize("locale", _TRANSLATED_LOCALES)
def test_translated_placeholders_match_english(locale):
    """A dropped or added ``{}`` is an IndexError at render time, not a typo.

    Many of these messages reach ``str.format`` with a fixed argument count, so
    a translator rewording one has to carry every placeholder across.
    """
    english, translated = _catalog("en"), _catalog(locale)
    mismatched = {
        key: (value.count("{}"), translated[key].count("{}"))
        for key, value in english.items()
        if value.count("{}") != translated[key].count("{}")
    }

    assert not mismatched, f"{locale} placeholder counts differ: {mismatched}"


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


def _catalog_chains(tree):
    """Every ``ms.a.b.c`` attribute chain in a module, as tuples of names.

    Also follows one level of aliasing (``tables = ms.analysis.tables``), which
    the widgets use heavily to keep call sites short.
    """
    aliases = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        chain = _chain_of(node.value, aliases)
        if isinstance(target, ast.Name) and chain:
            aliases[target.id] = chain

    chains = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(
            getattr(node, "parent", None), ast.Attribute
        ):
            chain = _chain_of(node, aliases)
            if chain:
                chains.add(chain)
    return chains


def _chain_of(node, aliases):
    """Resolve an attribute node to a catalog chain, or ``None``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    root = node.id
    if root == "ms":
        return tuple(reversed(parts))
    if root in aliases:
        return aliases[root] + tuple(reversed(parts))
    return None


def test_every_catalog_key_used_in_the_app_exists():
    """Guards typo'd ``ms.`` paths that only a rarely-rendered branch would hit.

    The catalog raises on a missing key, so an unrendered screen (the sample
    calculation tile, an error branch) would carry the break to production.
    """
    ms = get_translator()
    sources = [*(_REPO_ROOT / "component").rglob("*.py"), _REPO_ROOT / "app.py"]
    missing = set()
    for path in sources:
        tree = ast.parse(path.read_text())
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        for chain in _catalog_chains(tree):
            node = ms
            for i, part in enumerate(chain):
                if isinstance(node, str):
                    break  # the rest is a str method, e.g. .format
                if not isinstance(node, Box) or part not in node:
                    missing.add(f"{path.name}: ms.{'.'.join(chain[: i + 1])}")
                    break
                node = node[part]

    assert not missing, "catalog keys referenced but not defined: " + ", ".join(
        sorted(missing)
    )
