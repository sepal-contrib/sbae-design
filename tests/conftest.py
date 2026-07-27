import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    """Guarantee every test starts with a live main-thread event loop.

    Several tests drive coroutines with ``asyncio.run()``, which closes the loop
    it created and clears the thread's current loop. A later test that calls
    ``solara.render()`` (which reaches ``asyncio.get_event_loop()``) would then
    raise "There is no current event loop". Repairing before each test makes the
    suite order-independent.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield
