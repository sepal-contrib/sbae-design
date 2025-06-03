# Configure the logs

There's a file called `logging_config.example.toml` in the root of the project. To configure the logs, copy this file to `logging_config.toml`, the module will use that file as configuration.

There are two output streams configured by default: one for a file and one for a notebook console. You can change the configuration in the `logging_config.toml` file.

```toml
[loggers.sbae]
level     = "DEBUG"
handlers  = ["file", "notebook"] # Change this to ["file"] if you don't want notebook logs
propagate = false
```

# To view the logs in a file

You can view the logs in a file by running the following command in your terminal:

```bash
# cd into the project directory
tail -f sbae.log
```

# Using the logger

To use the logger in your code, you can import it as follows:

```python

import logging

# Get the logger instance
logger = logging.getLogger("sbae")

# Use the logger to log messages at different levels
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")


```

# Adding new packages

If your application needs custom packages you can add them to the `requirements.txt` file.
