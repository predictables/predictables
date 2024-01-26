from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Line(_BaseTraceHierarchyType):
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., width: Any | None = ..., **kwargs
    ) -> None: ...
