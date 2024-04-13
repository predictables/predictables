from .config import setup_logging, print_message_if_no_env_file

# Initialize logging
setup_logging()

# Check for .env file presence
print_message_if_no_env_file()
