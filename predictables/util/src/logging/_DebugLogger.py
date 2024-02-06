import logging as _logging
import os
import uuid as _uuid
from typing import Optional

import dotenv

from predictables.util.src.logging._LogLevel import LogLevel


class DebugLogger:
    """
    A class to log debug messages with a unique identifier to identify the debug session.
    """

    uuid: Optional[_uuid.UUID]
    turned_on: bool

    def __init__(self):
        """
        Initializes the DebugLogger class.
        """
        self.uuid = _uuid.uuid1()

        # Load the .env file to get the logging level - if we are not at the debug
        # level, we don't want to log anything.
        dotenv.load_dotenv()
        self.turned_on = (
            LogLevel.convert_str(os.getenv("LOGGING_LEVEL", "info").lower()) == "DEBUG"
        )

        if self.turned_on:
            self._init_log()

    def _init_log(self):
        """
        Initializes the logging module.
        """
        _logging.basicConfig(level=_logging.DEBUG)
        _logging.debug(f"Debugging UUID: {self.uuid}")

    def debug(self, message: str):
        """
        Logs a debug message with the unique identifier.

        Parameters
        ----------
        message : str
            The debug message to log.
        """
        if self.turned_on:
            _logging.debug(f"Debugging UUID: {self.uuid} - {message}")

    def msg(self, message: str):
        """
        Alias for the debug method.
        """
        self.debug(message)

    def turn_on(self):
        """
        Turns on the debug logger.
        """
        self.turned_on = True
        self._init_log()

    def turn_off(self):
        """
        Turns off the debug logger.
        """
        self.turned_on = False
        _logging.shutdown()
