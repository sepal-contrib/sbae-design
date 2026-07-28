"""``_TileLoopbackBridge`` enables jupyter_loopback by default (opt out with =0).

Each case runs in a subprocess: enabling the bridge mutates process-global
singletons that would otherwise leak across tests.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RENDER_SCRIPT = """
import os
_flag = {flag!r}
if _flag is None:
    os.environ.pop("LOCALTILESERVER_COMM_BRIDGE", None)
else:
    os.environ["LOCALTILESERVER_COMM_BRIDGE"] = _flag

import solara
import solara.server.kernel_context as kc
from sepal_ui.solara import setup_sessions

ctx = kc.create_dummy_context()
kc.set_current_context(ctx)
setup_sessions()

import app
import jupyter_loopback as jl

solara.render(app._TileLoopbackBridge(), handle_error=False)
print("ENABLED" if jl.is_comm_bridge_enabled() else "DISABLED")
"""


def _bridge_enabled(flag) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", _RENDER_SCRIPT.format(flag=flag)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    if "ENABLED" in result.stdout:
        return True
    if "DISABLED" in result.stdout:
        return False
    raise AssertionError(
        f"unexpected output: {result.stdout!r}\n{result.stderr[-2000:]}"
    )


def test_bridge_enabled_by_default():
    # the SEPAL case: flag unset, bridge must still mount
    assert _bridge_enabled(None) is True


def test_bridge_disabled_when_opted_out():
    assert _bridge_enabled("0") is False
