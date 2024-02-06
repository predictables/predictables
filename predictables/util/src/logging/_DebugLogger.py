import logging as _logging
import uuid as _uuid
from typing import Optional


class DebugLogger:
    """
    A class to log debug messages with a unique identifier to identify the debug session.
    """

    turned_on: bool
    uuid: Optional[_uuid.UUID]

    def __init__(self, turned_on: bool = False):
        """
        Initializes the DebugLogger class.

        Parameters
        ----------
        turned_on : bool, optional
            If True, the debug logger will be turned on, by default False. You must
            specifically set this to True to turn on the debug logger.

        Raises
        ------
        ValueError
            If the turned_on parameter is not a boolean.
        """
        if not isinstance(turned_on, bool):
            raise ValueError("The turned_on parameter must be a boolean.")
        self.uuid = _uuid.uuid1()

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
