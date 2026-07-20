"""Quiet the chatty tile-server access/debug logs.

The raster path (``localtileserver``) and the sample-points path
(``vectortileserver``) each run their own ``uvicorn`` instance. Both emit an
``INFO`` access line per tile request (hundreds while panning a map), and
``vectortileserver``'s ``"VECTORTILES"`` logger runs at ``DEBUG``.

``uvicorn`` re-applies its own logging config each time a server starts, so
setting these levels once at import does not stick. Call
:func:`quiet_tile_server_logs` right AFTER a tile server comes up (i.e. just
after constructing a ``TileClient``); it is idempotent, so calling it per client
is fine and keeps the level pinned no matter which server booted last.
"""

import logging

# uvicorn's shared access/error loggers (process-global across every uvicorn
# instance), vectortileserver's "VECTORTILES" logger, and the two libraries'
# module loggers.
_NOISY_LOGGERS = (
    "uvicorn.access",
    "uvicorn.error",
    "VECTORTILES",
    "vectortileserver",
    "localtileserver",
)


def quiet_tile_server_logs(level: int = logging.WARNING) -> None:
    """Raise the tile-server loggers to ``level`` (default WARNING).

    Suppresses per-tile uvicorn access lines and vectortileserver's DEBUG
    chatter while leaving real warnings/errors visible.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
