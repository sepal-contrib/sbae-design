# component/logger_config.py

import logging
import os
from pathlib import Path

LOG_FILE_NAME = "sampling_tool.log"
LOG_APP_NAME = "sampling_tool_app" # Parent logger name

def setup_logging(base_dir=None, log_level_file=logging.DEBUG, log_level_console=logging.INFO, log_to_console=False):
    """
    Configures logging for the application.

    Args:
        base_dir (str, optional): Base directory for the 'logs' folder. Defaults to os.getcwd().
        log_level_file (int, optional): Logging level for the file handler. Defaults to logging.DEBUG.
        log_level_console (int, optional): Logging level for the console handler. Defaults to logging.INFO.
        log_to_console (bool, optional): Whether to also output logs to the console. Defaults to False.
    """
    if base_dir is None:
        base_dir = Path(os.getcwd()) # Default to current working directory
    else:
        base_dir = Path(base_dir)
    
    log_directory = base_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file_path = log_directory / LOG_FILE_NAME

    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    
    # Get the application's parent logger
    app_logger = logging.getLogger(LOG_APP_NAME)
    app_logger.setLevel(min(log_level_file, log_level_console if log_to_console else log_level_file)) # Set logger to the lowest level of its handlers
    
    # Remove existing handlers to avoid duplication if setup is called multiple times (e.g. in Jupyter)
    if app_logger.hasHandlers():
        app_logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode='w') # Write mode
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level_file)
    app_logger.addHandler(file_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(log_level_console)
        app_logger.addHandler(console_handler)

    # Initial log message to confirm setup
    # Use a generic logger for this initial message to avoid depending on LOG_APP_NAME being set up for itself.
    # Or, if certain this runs only once for the app_logger:
    initial_logger = logging.getLogger(f"{LOG_APP_NAME}.config") # Child logger for this message
    initial_logger.info(f"Logging configured. Log level for file: {logging.getLevelName(log_level_file)}. Log file: {log_file_path}")
    if log_to_console:
        initial_logger.info(f"Console logging enabled at level: {logging.getLevelName(log_level_console)}.")