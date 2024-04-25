# from ._PredicTables import PredicTables  # noqa: ERA001
# from .src import *  # noqa: ERA001

from .config import print_message_if_no_env_file
from predictables.util.logging import get_logger

# Check for .env file presence
print_message_if_no_env_file()

# Initialize logging
logger = get_logger()
