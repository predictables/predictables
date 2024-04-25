"""Set up the logging configuration and check for the presence of a .env file."""

from dotenv import load_dotenv
from pathlib import Path
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
