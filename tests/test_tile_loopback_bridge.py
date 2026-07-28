"""The tile comm-bridge mount enables jupyter_loopback when flagged.

``_TileLoopbackBridge`` mounts jupyter_loopback's anywidget comm bridge so
localtileserver's ``http://127.0.0.1:<port>`` tile URLs reach the browser when
the app is served behind a reverse proxy (``run-solara --serve`` / SEPAL). It is
gated on ``LOCALTILESERVER_COMM_BRIDGE`` so plain local runs keep direct-HTTP
tiles. This is a Python-side smoke test: it confirms the component renders and
enables the bridge when the flag is set (the actual browser interception is only
observable in a live frontend).

Runs in a subprocess (like ``test_app_page_render``) because enabling the bridge
and rendering under a Solara kernel context mutate process-global singletons.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RENDER_SCRIPT = """
import os
os.environ["LOCALTILESERVER_COMM_BRIDGE"] = "1"

import solara
import solara.server.kernel_context as kc
from sepal_ui.solara import setup_sessions

ctx = kc.create_dummy_context()
kc.set_current_context(ctx)
setup_sessions()

import app
import jupyter_loopback as jl

solara.render(app._TileLoopbackBridge(), handle_error=False)
assert jl.is_comm_bridge_enabled(), "comm bridge was not enabled with the flag set"
print("BRIDGE_OK")
"""


def test_bridge_enables_when_flagged():
    result = subprocess.run(
        [sys.executable, "-c", _RENDER_SCRIPT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "BRIDGE_OK" in result.stdout
