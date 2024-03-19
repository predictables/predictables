from __future__ import annotations

import datetime
import inspect
import json
import sys
import uuid
from pathlib import Path
from predictables.util.src.logging._LogLevel import LogLevel


class Log:
    def __init__(self):
        self.log = {
            "log_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "cwd": Path.cwd(),
            "source_file": sys.argv[0],
            "line_number": self._get_line_number(),
            "message": "",
            "level": LogLevel.INFO.value,
        }

    def _get_line_number(self) -> int:
        """Return the line number the logger was called from."""
        return inspect.currentframe().f_back.f_lineno

    def _msg(self, message: str, level: str | int | LogLevel = LogLevel.INFO) -> Log:
        if isinstance(level, str):
            level = LogLevel.from_str(level)
        elif isinstance(level, int):
            level = LogLevel.from_int(level)
        self.log["message"] = message
        self.log["level"] = level.value
        return self

    def info(self, msg: str) -> Log:
        return self._msg(msg, LogLevel.INFO)

    def debug(self, msg: str) -> Log:
        return self._msg(msg, LogLevel.DEBUG)

    def warning(self, msg: str) -> Log:
        return self._msg(msg, LogLevel.WARNING)

    def error(self, msg: str) -> Log:
        return self._msg(msg, LogLevel.ERROR)

    def critical(self, msg: str) -> Log:
        return self._msg(msg, LogLevel.CRITICAL)

    def json(self) -> str:
        return json.dumps(self.log)
