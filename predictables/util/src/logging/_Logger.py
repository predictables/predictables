import json
import os
from typing import Optional

from predictables.util.src.logging._Log import Log


class Logger:
    def __init__(
        self,
        name: str,
        file_name: Optional[str] = None,
    ):
        """
        Create a logger that logs _Log

        Parameters
        ----------
        name : str
            The name of the logger. This is the only required parameter.
        format_str : str, optional
            The format of the log messages. The default is "%(asctime)s - %(name)s - %(levelname)s - %(message)s".

        Returns
        -------
        None. Initializes the logger, but need to call `get_logger()` to actually get the logger.

        """
        self.name = name
        self.file_name = (
            file_name if file_name else f"{name.lower().replace(' ', '_')}.log"
        )

        # Create the log file if it doesn't exist (will be json)
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w") as f:
                f.write("[]")

    def add(self, msg, level, *args, **kwargs):
        """
        Add a json log message to the logger.

        Parameters
        ----------
        msg : str
            The message to log.
        level : str
            The level of the log message. Must be one of "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL".

        Returns
        -------
        None. Adds a log message to the log file.
        """
        log = Log().info(msg)
        with open(self.file_name, "r") as f:
            logs = json.load(f)
        logs.append(log.json())
        with open(self.file_name, "w") as f:
            json.dump(logs, f)

    def info(self, msg, *args, **kwargs):
        """
        Add an info log message to the logger.

        Parameters
        ----------
        msg : str
            The message to log.

        Returns
        -------
        None. Adds a log message to the log file.
        """
        self.add(Log().info(msg).json(), "INFO")

    def debug(self, msg, *args, **kwargs):
        """
        Add a debug log message to the logger.

        Parameters
        ----------
        msg : str
            The message to log.

        Returns
        -------
        None. Adds a log message to the log file.
        """
        self.add(Log().debug(msg).json(), "DEBUG")

    def warning(self, msg, *args, **kwargs):
        """
        Add a warning log message to the logger.

        Parameters
        ----------
        msg : str
            The message to log.

        Returns
        -------
        None. Adds a log message to the log file.
        """
        self.add(Log().warning(msg).json(), "WARNING")

    def error(self, msg, *args, **kwargs):
        """
        Add an error log message to the logger.

        Parameters
        ----------
        msg : str
            The message to log.

        Returns
        -------
        None. Adds a log message to the log file.
        """
        self.add(Log().error(msg).json(), "ERROR")

    def critical(self, msg, *args, **kwargs):
        """
        Add a critical log message to the logger.

        Parameters
        ----------
        msg : str
            The message to log.

        Returns
        -------
        None. Adds a log message to the log file.
        """
        self.add(Log().critical(msg).json(), "CRITICAL")
