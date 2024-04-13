"""Set up the logging configuration and check for the presence of a .env file."""

import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from predictables.util.src.logging._LogLevel import LogLevel
import warnings

# Load environment variables at the beginning
load_dotenv()


def print_message_if_no_env_file() -> None:
    """Print a warning message if no .env file is found."""
    if not Path(".env").exists():
        msg = """No .env file found. Using default values for:
    - LOGGING_LEVEL: INFO
    - LOGGING_FILE: predictables.log
    - LOGGING_MODE: w

    - TQDM_ENABLE: true
    - TQDM_NOTEBOOK: false

If these configuration values are ok, you can ignore this message. The impact will not touch the modeling itself, but only things like logging and display of progress bars, etc.

To suppress this message and/or set custom values for these variables, create a .env file in the root directory of your project.
    """
        warnings.warn(msg, UserWarning, stacklevel=2)


def get_logging_level() -> int:
    """Get the logging level from the environment."""
    level_str = os.getenv("LOGGING_LEVEL", "INFO").upper()
    return LogLevel.from_str(level_str).value


def get_logfile_name() -> str:
    """Get the name of the log file."""
    return os.getenv("LOGGING_FILE", "predictables.log")


def get_logfile_mode() -> str:
    """Get the mode for the log file."""
    return os.getenv("LOGGING_MODE", "w")


def setup_logging() -> None:
    """Configure the logging settings."""
    logging.basicConfig(
        level=get_logging_level(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=get_logfile_name(),
        filemode=get_logfile_mode(),
    )
