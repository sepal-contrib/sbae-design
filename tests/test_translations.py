"""The message catalog and the locale wiring behind it.

pysepal 4 resolves the locale in the browser and holds it in one Solara
reactive per kernel; ``msg()`` reads it on every call. These guard the two
halves: every shipped catalog must bind cleanly, and a language change must
reach a mounted component.
"""

import ast
import json
from pathlib import Path
from string import Formatter

import ipyvuetify as v
import pytest
from pysepal.i18n import MissingMessageError, catalog, set_locale

from component.message import MESSAGE_DIR, messages, msg
from component.widget.aoi_upload_selector import UploadDialogCard

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLURAL_CATEGORIES = {"zero", "one", "two", "few", "many", "other"}


@pytest.fixture(autouse=True)
def _reset_locale():
    """Keep a locale switch from leaking into the rest of the suite."""
    yield
    set_locale("en")


def _button_labels(rc) -> str:
    return " ".join(str(c) for b in rc.find(v.Btn).widgets for c in (b.children or []))


def test_translated_catalogs_match_english():
    """Missing keys, orphans, placeholder and plural-shape drift all show here.

    Each would otherwise fall back to English silently, so none is visible in
    the UI.
    """
    assert messages.check() == ()


def test_every_shipped_locale_renders():
    """``available_locales`` feeds the picker, so every folder must render."""
    for locale in messages.available_locales():
        set_locale(locale)
        assert msg("app.title")


def test_available_locales_are_the_catalog_folders_only():
    """``__pycache__`` sits beside the catalogs and must never be offered."""
    assert messages.available_locales() == ("en", "es", "fr")


def test_untranslated_key_falls_back_to_english(tmp_path):
    """A key a target catalog omits still renders, in English.

    Built against a synthetic catalog rather than the shipped ones so it keeps
    guarding the fallback once es/fr are complete.
    """
    (tmp_path / "en").mkdir()
    (tmp_path / "de").mkdir()
    (tmp_path / "en" / "locale.json").write_text(
        json.dumps({"a": {"translated": "one", "untranslated": "two"}})
    )
    (tmp_path / "de" / "locale.json").write_text(
        json.dumps({"a": {"translated": "eins"}})
    )
    synthetic = catalog(tmp_path)

    set_locale("de")

    assert synthetic.msg("a.translated") == "eins"
    assert synthetic.msg("a.untranslated") == "two"


def test_browser_locale_variant_resolves_to_a_shipped_catalog():
    """``navigator.language`` reports es-CL where the app only ships ``es``."""
    set_locale("es-CL")
    assert msg("common.close") == "Cerrar"


def test_unknown_locale_resolves_to_english():
    set_locale("de")
    assert msg("common.close") == "Close"


def test_plural_nodes_follow_each_locale_rules():
    assert msg("design.stats.class_count", count=1) == "1 class"
    assert msg("design.stats.class_count", count=3) == "3 classes"
    set_locale("fr")
    assert msg("design.stats.class_count", count=0) == "0 classe"


def test_missing_key_raises_instead_of_rendering_blank():
    """A typo'd key must fail loudly rather than render an empty string."""
    with pytest.raises(MissingMessageError):
        msg("upload.no_such_key")


def test_language_change_re_renders_a_mounted_component(render_in_app):
    """Switching locale re-renders in place -- no reload, no remount."""
    _, rc = render_in_app(
        lambda: UploadDialogCard(sbae_map=None, on_close=lambda: None)
    )
    assert "Close" in _button_labels(rc)

    set_locale("es")

    assert "Cerrar" in _button_labels(rc)


def _english_nodes():
    """English as ``{dotted key: str | plural dict | subtree dict}``."""
    nodes = {}

    def walk(node, prefix):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            nodes[path] = value
            if isinstance(value, dict) and not set(value) <= _PLURAL_CATEGORIES:
                walk(value, path)

    walk(json.loads((MESSAGE_DIR / "en" / "locale.json").read_text()), "")
    return nodes


def _fields(template):
    return {field for _, field, _, _ in Formatter().parse(template) if field}


def _msg_calls(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "msg"
            and node.args
        ):
            yield node


def test_every_msg_call_names_a_key_and_its_placeholders():
    """Guards keys and arguments that only a rarely-rendered branch would hit.

    ``msg()`` raises on an unknown key or a missing placeholder, so an error
    branch nobody renders in development would carry the break to production.
    """
    english = _english_nodes()
    sources = [*(_REPO_ROOT / "component").rglob("*.py"), _REPO_ROOT / "app.py"]
    problems = []
    for path in sources:
        for call in _msg_calls(ast.parse(path.read_text())):
            where = f"{path.name}:{call.lineno}"
            key_node = call.args[0]
            if isinstance(key_node, ast.JoinedStr):
                prefix = key_node.values[0]
                if not (
                    isinstance(prefix, ast.Constant)
                    and isinstance(english.get(prefix.value.rstrip(".")), dict)
                ):
                    problems.append(f"{where}: computed key has no catalog subtree")
                continue
            if not isinstance(key_node, ast.Constant):
                continue
            node = english.get(key_node.value)
            if node is None:
                problems.append(f"{where}: no message {key_node.value!r}")
                continue
            given = {kw.arg for kw in call.keywords}
            if None in given:
                continue  # **values: cannot check statically
            if isinstance(node, dict):
                if not set(node) <= _PLURAL_CATEGORIES:
                    problems.append(f"{where}: {key_node.value!r} is not a message")
                    continue
                wanted = set().union(*map(_fields, node.values())) | {"count"}
            else:
                wanted = _fields(node)
            if given != wanted:
                problems.append(
                    f"{where}: {key_node.value!r} takes {sorted(wanted)}, "
                    f"got {sorted(given)}"
                )

    assert not problems, "\n".join(problems)
