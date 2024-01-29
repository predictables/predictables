from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Connector(_BaseTraceHierarchyType):
    @property
    def fillcolor(self): ...
    @fillcolor.setter
    def fillcolor(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        fillcolor: Any | None = ...,
        line: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...
