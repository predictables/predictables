import logging
import os
from typing import Optional


class Logger:
    def __init__(
        self,
        name: str,
        file_name: Optional[str] = None,
        level: int = logging.DEBUG,
        format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        date_fmt_str: str = "%Y-%m-%d",
    ):
        """
        Create a logger that logs to the console and to a file.

        Parameters
        ----------
        name : str
            The name of the logger. This is the only required parameter.
        file_name : str, optional
            The name of the file to log to. If not provided, the file will be named self.name.log.
        level : int, optional
            The logging level. The default is logging.DEBUG.
        format_str : str, optional
            The format of the log messages. The default is "%(asctime)s - %(name)s - %(levelname)s - %(message)s".
        date_fmt_str : str, optional
            The format of the date in the log messages. The default is "%Y-%m-%d".

        Returns
        -------
        None. Initializes the logger, but need to call `get_logger()` to actually get the logger.

        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create formatter
        formatter = logging.Formatter(format_str)
        formatter.datefmt = date_fmt_str
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

        # export log to file
        if file_name is None:
            # get the folder of the module calling this function
            file_name = os.path.dirname(os.path.realpath(__file__))

        # add a folder called logs if it doesn't exist
        if not os.path.exists(f"{file_name}/logs"):
            os.makedirs(f"{file_name}/logs")

        # add a file called self.name.log if it doesn't exist
        file_name = f"{file_name}/logs/{name}.log"

        fh = logging.FileHandler(file_name, mode="w")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
