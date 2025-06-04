"""Logging configuration for the Seplan project.

To use this logging configuration, set the environment variable
SBAE_LOG_CFG to the path of the logging configuration file.
The repo has a sample configuration file in the root directory.

"""

import logging
import logging.config
import os
from pathlib import Path

import tomli


def setup_logging():
    cfg_path = (
        os.getenv("SBAE_LOG_CFG")
        or Path(__file__).parent.parent.parent / "logging_config.toml"
    )

    cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        sbae_logger = logging.getLogger("sbae")
        for handler in sbae_logger.handlers[:]:
            sbae_logger.removeHandler(handler)
        sbae_logger.addHandler(logging.NullHandler())
        return

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Logging config not found at {cfg_path}")

    with cfg_path.open("rb") as f:
        cfg = tomli.load(f)

    logging.config.dictConfig(cfg)
