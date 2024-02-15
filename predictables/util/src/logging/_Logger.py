import datetime
import json
import os
import uuid
from typing import Optional

from predictables.util.src.logging._Log import Log


class Logger:
    def __init__(
        self,
        name: str = "log",
        file_name: Optional[str] = None,
    ):
        """
        Create a logger that logs _Log

        Parameters
        ----------
        name : str
            The name of the logger. This is the only required parameter.
        format_str : str, optional
            The format of the log messages. The default is
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s".

        Returns
        -------
        None. Initializes the logger, but need to call `get_logger()` to actually get
        the logger.

        """
        self.name = name
        self.file_name = (
            file_name if file_name else f"{name.lower().replace(' ', '_')}.json"
        )
        self.session_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_id = str(uuid.uuid4())

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
            The level of the log message. Must be one of "INFO", "DEBUG", "WARNING",
            "ERROR", "CRITICAL".

        Returns
        -------
        None. Adds a log message to the log file.
        """
        log = Log().info(msg)
        log.log["session_id"] = self.session_id
        log.log["session_ts"] = self.session_ts

        # Open the log file and append the log message
        with open(self.file_name, "rb") as f:
            logs = json.loads(f)

        # Extract the log from the string
        log = json.loads(log)

        # Make sure the log is a dict, and logs is a list
        if not isinstance(logs, list):
            logs = [logs]
        if not isinstance(log, dict):
            log = log.log

        logs.append(log)

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
