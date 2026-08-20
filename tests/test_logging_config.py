"""quiet_tile_server_logs raises the chatty tile-server loggers to WARNING."""

import logging

from component.scripts.logging_config import _NOISY_LOGGERS, quiet_tile_server_logs


def test_quiet_tile_server_logs_sets_warning():
    saved = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    try:
        quiet_tile_server_logs()
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING, name
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def test_quiet_tile_server_logs_covers_the_noisy_sources():
    # The two access-log + debug-log sources seen in the served terminal.
    assert "uvicorn.access" in _NOISY_LOGGERS
    assert "VECTORTILES" in _NOISY_LOGGERS
