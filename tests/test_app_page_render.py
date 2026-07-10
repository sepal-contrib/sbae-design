"""Regression: the full Page must render under a Solara kernel context.

A bare ``solara.render()`` misses this class of failure because
``@solara.lab.on_kernel_start`` (which activates pysepal's SessionManager) only
fires under a live kernel. Once the session manager is active, pysepal's MapApp
requires a per-kernel ThemeState; ``app.Page`` provides one explicitly.

This runs in a subprocess: it activates the session manager and renders the
real ``Page`` inside a virtual kernel context, which mutates process-global
state (singletons, kernel context) that would otherwise leak into other tests.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RENDER_SCRIPT = """
import solara
import solara.server.kernel_context as kc
from sepal_ui.solara import setup_sessions

ctx = kc.create_dummy_context()
kc.set_current_context(ctx)
setup_sessions()  # activates SessionManager, exactly as the server does
import app

solara.render(app.Page(), handle_error=False)
print("PAGE_RENDER_OK")
"""

_WIDGET_RENDER_SCRIPT = """
import solara
import solara.server.kernel_context as kc
import app

solara.render(app.Page(), handle_error=False)
assert not kc.has_current_context()
print("PAGE_WIDGET_RENDER_OK")
"""


def _render_env(tmp_path):
    return {
        **os.environ,
        "HOME": str(tmp_path),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }


def test_page_renders_under_kernel_context(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _RENDER_SCRIPT],
        cwd=str(_REPO_ROOT),
        env=_render_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-3000:]
    assert "PAGE_RENDER_OK" in result.stdout


def test_page_renders_without_solara_server_context(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _WIDGET_RENDER_SCRIPT],
        cwd=str(_REPO_ROOT),
        env=_render_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-3000:]
    assert "PAGE_WIDGET_RENDER_OK" in result.stdout
