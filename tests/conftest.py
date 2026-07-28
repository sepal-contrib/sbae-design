import asyncio

import pytest


@pytest.fixture(autouse=True)
def _fresh_event_loop():
    """Give every test its own live main-thread event loop.

    Several tests drive coroutines with ``asyncio.run()``, which closes the loop
    it created and clears the thread's current loop. A later test that calls
    ``solara.render()`` (reaching ``asyncio.get_event_loop()``) would then raise
    "There is no current event loop". Installing a fresh loop per test makes the
    suite order-independent without touching the deprecated ``get_event_loop``.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        if not loop.is_closed():
            loop.close()
