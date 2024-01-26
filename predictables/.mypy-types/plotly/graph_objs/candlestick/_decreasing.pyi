from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Decreasing(_BaseTraceHierarchyType):
    @property
    def fillcolor(self): ...
    @fillcolor.setter
    def fillcolor(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        fillcolor: Any | None = ...,
        line: Any | None = ...,
        **kwargs
    ) -> None: ...
