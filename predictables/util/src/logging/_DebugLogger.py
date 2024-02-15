import datetime
import logging as _logging
import os
import uuid as _uuid
from typing import Callable, Optional

from dotenv import load_dotenv

from predictables.util.src.logging._LogLevel import LogLevel

load_dotenv()


class DebugLogger(_logging.Logger):
    """
    A class to log debug messages with a unique identifier to identify the debug session.
    """

    uuid: Optional[_uuid.UUID]
    turned_on: bool

    def __init__(
        self,
        filename: str = "debug.log",
        working_file: Optional[str] = None,
        message_prefix: Optional[Callable] = None,
    ):
        """
        Initializes the DebugLogger class.
        """
        self.uuid = _uuid.uuid1()

        # Load the .env file to get the logging level - if we are not at the debug
        # level, we don't want to log anything.
        self.turned_on = (
            LogLevel.convert_str(os.getenv("LOGGING_LEVEL", "info").lower())
            == "DEBUG"
        )

        if self.turned_on:
            self.filename = filename
            self._init_log()
            self.level = LogLevel.DEBUG  # type: ignore

        super().__init__(str(self.uuid))
        self.working_file = working_file
        self.message_prefix = (
            message_prefix
            if message_prefix is not None
            else self._default_message_prefix
        )

    def _default_message_prefix(self) -> str:
        return f"{datetime.datetime.now()} - {self.uuid} - {self.working_file}"

    def _init_log(self):
        """
        Initializes the logging module.
        """
        _logging.basicConfig(filename=self.filename, level=_logging.DEBUG)
        _logging.debug(f"Debugging UUID: {self.uuid}")

    def debug_(self, message: str):
        """
        Logs a debug message with the unique identifier.

        Parameters
        ----------
        message : str
            The debug message to log.
        """
        if self.turned_on:
            _logging.debug(f"{self._default_message_prefix()} - {message}")

    def msg(self, message: str):
        """
        Alias for the debug method.
        """
        self.debug_(message)

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
