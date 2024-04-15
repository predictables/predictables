from .config import print_message_if_no_env_file
from .logging import get_logger

# Check for .env file presence
print_message_if_no_env_file()

# Initialize logging
logger = get_logger()