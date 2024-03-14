from enum import Enum


class LogLevel(Enum):
    """
    Enum class for logging levels.

    Attributes
    ----------
    INFO : str
        The info logging level.
    DEBUG : str
        The debug logging level.
    WARNING : str
        The warning logging level.
    ERROR : str
        The error logging level.
    CRITICAL : str
        The critical logging level.

    Methods
    -------
    from_str(level: str)
        Returns the LogLevel enum from a string. Accepts one (case insensitive) of:
            1. First letter of the level (I, D, W, E, C)
            2. The full level name (INFO, DEBUG, WARNING, ERROR, CRITICAL)
            3. A string with the integer value of the level (1, 2, 3, 4, 5)
    _str(level: str)
        Alias for from_str. Returns the LogLevel enum from a string.
    from_int(level: int)
        Returns the LogLevel enum from an integer. Accepts an integer between 1 and 5.
        If the integer is less than 1, returns INFO. If the integer is greater than 5,
        returns CRITICAL.
    _int(level: int)
        Alias for from_int. Returns the LogLevel enum from an integer.
    get_str()
        Returns the string representation of the LogLevel enum.
    str_()
        Alias for get_str. Returns the string representation of the LogLevel enum.
    get_int(level)
        Returns the integer representation of the LogLevel enum.
    int_(level)
        Alias for get_int. Returns the integer representation of the LogLevel enum.

    """

    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def from_str(cls, level: str) -> "LogLevel":
        mapping = {
            "I": "INFO",
            "INFO": "INFO",
            "D": "DEBUG",
            "DEBUG": "DEBUG",
            "W": "WARNING",
            "WARNING": "WARNING",
            "E": "ERROR",
            "ERROR": "ERROR",
            "C": "CRITICAL",
            "CRITICAL": "CRITICAL",
            "1": "INFO",
            "2": "DEBUG",
            "3": "WARNING",
            "4": "ERROR",
            "5": "CRITICAL",
        }

        return cls[mapping[level.upper()]]

    @classmethod
    def _str(cls, level: str) -> "LogLevel":
        return cls.from_str(level)

    @classmethod
    def from_int(cls, level: int) -> "LogLevel":
        threshold = 5
        if level > threshold:
            return cls.from_str("CRITICAL")
        elif level < 1:
            return cls.from_str("INFO")
        else:
            return cls(list(cls)[level - 1])

    @classmethod
    def _int(cls, level: int) -> "LogLevel":
        return cls.from_int(level)

    def get_str(self) -> str:
        return self.name

    def str_(self) -> str:
        return self.get_str()

    def get_int(self, level: "LogLevel") -> int:
        return level.value

    def int_(self, level: "LogLevel") -> int:
        return self.get_int(level)

    @staticmethod
    def convert_str(level: str) -> str:
        return LogLevel.from_str(level).get_str()

    @staticmethod
    def convert_int(level: int) -> str:
        return LogLevel.from_int(level).get_str()
