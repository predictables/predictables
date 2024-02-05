import datetime
import json
import os
import sys
from typing import Union

from predictables.util.src.logging._LogLevel import LogLevel


class Log:
    def __init__(self):
        self.log = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "cwd": os.getcwd(),
            "source_file": sys.argv[0],
            "line_number": 0,
            "message": "",
            "level": LogLevel.INFO.value,
        }
        return self

    def _get_line_number(self):
        """Returns the line number the logger was called from."""

    def _msg(self, message: str, level: Union[str, int, LogLevel] = LogLevel.INFO):
        if isinstance(level, str):
            level = LogLevel.from_str(level)
        elif isinstance(level, int):
            level = LogLevel.from_int(level)
        self.log["message"] = message
        self.log["level"] = level.value
        return self

    def info(self, msg):
        return self._msg(msg, LogLevel.INFO)

    def debug(self, msg):
        return self._msg(msg, LogLevel.DEBUG)

    def warning(self, msg):
        return self._msg(msg, LogLevel.WARNING)

    def error(self, msg):
        return self._msg(msg, LogLevel.ERROR)

    def critical(self, msg):
        return self._msg(msg, LogLevel.CRITICAL)

    def json(self):
        return json.dumps(self.log)
