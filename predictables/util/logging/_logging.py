"""Set up logging configuration."""

import logging
import os
from dotenv import load_dotenv
from predictables.util.logging._LogLevel import LogLevel

# Load environment variables at the beginning
load_dotenv()


def get_logging_level() -> int:
    """Get the logging level from the environment."""
    level_str = os.getenv("LOGGING_LEVEL", "INFO").upper()
    return int(LogLevel.from_str(level_str).to_int())


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


def get_logger() -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger("predictables")
    handler = logging.FileHandler(get_logfile_name(), mode=get_logfile_mode())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(get_logging_level())
    return logger
