# Environment Setup Instructions

## In SEPAL

If you are developing this module inside a SEPAL environment, it is highly recommended to create a dedicated virtual environment for your work. This makes it easier to install and test packages specifically for this project without any dependency conflicts.

### 1. Creating the virtual environment (first time only)

1.  Navigate to your project's root directory:
    ```bash
    cd /path/to/your/project
    ```
2.  Run:
    ```bash
    module_venv
    ```
    - **What this does:**
      - Creates a fresh, local Python environment under  
        `/home/sepal-user/module-venv/sbae-design`
      - This new environment is structured to mirror the actual production environment as closely as possible, but unlike the read-only SEPAL production environment, you can freely modify this local version without worrying about permissions.
    - **Important:**
      - Because it installs everything from scratch, the initial setup may take **several minutes**. Please be patient—once it finishes, you'll have a self-contained environment that you can safely modify.

### 2. Activating the environment in Jupyter notebooks

Once your virtual environment exists, you'll want to run and test your notebooks inside it. To do so:

1.  Open JupyterLab (or Jupyter Notebook) from SEPAL.
2.  In the "Kernel" (or "Kernel → Change Kernel") menu, look for a kernel named:
    ```
    (test) sbae-design
    ```
    - That "(test) sbae-design" entry corresponds to the virtual environment you created with `module_venv`.
    - Selecting it will ensure any code you run within your notebooks uses the packages and Python version from your local `sbae-design` environment—**not** the global SEPAL Python.

### 3. Installing or updating packages later

Whenever you need to install a new package (or update an existing one) inside your `sbae-design` environment, follow these steps:

1.  From any SEPAL terminal, run:
    ```bash
    activate_venv
    ```
2.  Wait a few seconds. You will be prompted with a numbered list of all existing virtual environments on this SEPAL instance. For example:
    ```
    1) /home/sepal-user/module-venv/sbae-design
    2) /home/sepal-user/module-venv/another-project
    3) /home/sepal-user/module-venv/yet-another
    Select environment [1-3]:
    ```
3.  Type the number corresponding to `sbae-design` (e.g., `1`) and press Enter. You should then see something like:
    ```
    Activating: /home/sepal-user/module-venv/sbae-design
    (sbae-design) sepal-user@sepal:~$
    ```
    - At that point, your prompt is prefixed with `(sbae-design)`, indicating that the virtual environment is active.
4.  Now install or update any package, for example:
    ```bash
    pip install --upgrade sepal_ui
    ```
    - This will install (or upgrade) `sepal_ui` **inside** `sbae-design` only.
5.  Once you've confirmed the packages and versions work correctly, make sure to update the `requirements.txt` file. This file will be read by the SEPAL production environment to create the read-only environment that will be shared with other users.
6.  If you want to run notebooks again using your updated environment, go back into Jupyter and pick the "(test) sbae-design" kernel.

---

That's it! With this workflow, you'll have a reproducible, isolated environment for development—closely matching production but safely contained on your SEPAL instance.

# Configure the logs

There's a file called `logging_config.example.toml` in the root of the project. To configure the logs, copy this file to `logging_config.toml`, the module will use that file as configuration. The `logging_config.toml` file is already added to `.gitignore` to prevent it from being committed to the repository.

> **Important for Production Environments**: In production deployments, the `logging_config.toml` file should **NOT** exist by default. This ensures that no logging occurs in production.

There are two output streams configured by default: one for a file and one for a notebook console. You can change the configuration in the `logging_config.toml` file.

```toml
[loggers.sbae]
level     = "DEBUG"
handlers  = ["file", "notebook"] # Change this to ["file"] if you don't want notebook logs
propagate = false
```

## To view the logs in a file

You can view the logs in a file by running the following command in your terminal:

```bash
# cd into the project directory
tail -f sbae.log
```

## Using the logger

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
