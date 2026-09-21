"""Regression: the full Page must render under a Solara kernel context.

A bare ``solara.render()`` misses this class of failure because
``@solara.lab.on_kernel_start`` (which activates pysepal's SessionManager) only
fires under a live kernel. Once the session manager is active, pysepal's MapApp
requires a per-kernel ThemeState; ``app.Page`` provides one explicitly.

This runs in a subprocess: it activates the session manager and renders the
real ``Page`` inside a virtual kernel context, which mutates process-global
state (singletons, kernel context) that would otherwise leak into other tests.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RENDER_SCRIPT = """
import solara
import solara.server.kernel_context as kc
from pysepal.solara import setup_sessions

ctx = kc.create_dummy_context()
kc.set_current_context(ctx)
setup_sessions()  # activates SessionManager, exactly as the server does
import app

solara.render(app.Page(), handle_error=False)
print("PAGE_RENDER_OK")
"""

# Page reads its titles through msg(), so a language change re-renders it. The
# map is a stateful widget: rebuilt there, it comes back empty and every layer
# already drawn is lost for good, since nothing re-adds a layer it already added.
_LOCALE_SCRIPT = """
import solara
import solara.server.kernel_context as kc
from pysepal.solara import setup_sessions

ctx = kc.create_dummy_context()
kc.set_current_context(ctx)
setup_sessions()
import app
from ipyleaflet import Marker
from pysepal.i18n import set_locale
from pysepal.sepalwidgets.vue_app import MapApp

_, rc = solara.render(app.Page(), handle_error=False)
before = rc.find(MapApp).widget.main_map[0]
before.add_layer(Marker(location=(0, 0), name="kept"))

set_locale("es")

after = rc.find(MapApp).widget.main_map[0]
assert after is before, "the language change rebuilt the map"
assert "kept" in [layer.name for layer in after.layers]
print("MAP_KEPT_OK")
"""


def _run(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_page_renders_under_kernel_context():
    result = _run(_RENDER_SCRIPT)

    assert result.returncode == 0, result.stderr[-3000:]
    assert "PAGE_RENDER_OK" in result.stdout


def test_language_change_keeps_the_map_and_its_layers():
    result = _run(_LOCALE_SCRIPT)

    assert result.returncode == 0, result.stderr[-3000:]
    assert "MAP_KEPT_OK" in result.stdout
