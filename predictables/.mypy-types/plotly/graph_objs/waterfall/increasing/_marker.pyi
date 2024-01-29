from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Marker(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        line: Any | None = ...,
        **kwargs
    ) -> None: ...
