import logging as _logging
import os
import uuid as _uuid
from typing import Optional

from dotenv import load_env

from predictables.util.src.logging._LogLevel import LogLevel


class DebugLogger:
    """
    A class to log debug messages with a unique identifier to identify the debug session.
    """

    uuid: Optional[_uuid.UUID]

    def __init__(self):
        """
        Initializes the DebugLogger class.
        """
        self.uuid = _uuid.uuid1()
        log_level = LogLevel(os.getenv("LOGGING_LEVEL", False))
        self.turned_on = log_level.

        if turned_on:
            self.turned_on = True
            self._init_log()
        else:
            self.turned_on = False

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
