from typing import Any

class DegenerateDataWarning(RuntimeWarning):
    args: Any
    def __init__(self, msg: Any | None = ...) -> None: ...

class ConstantInputWarning(DegenerateDataWarning):
    args: Any
    def __init__(self, msg: Any | None = ...) -> None: ...

class NearConstantInputWarning(DegenerateDataWarning):
    args: Any
    def __init__(self, msg: Any | None = ...) -> None: ...

class FitError(RuntimeError):
    args: Any
    def __init__(self, msg: Any | None = ...) -> None: ...
