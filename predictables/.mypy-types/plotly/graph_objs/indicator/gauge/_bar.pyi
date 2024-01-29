from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Bar(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def thickness(self): ...
    @thickness.setter
    def thickness(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        line: Any | None = ...,
        thickness: Any | None = ...,
        **kwargs
    ) -> None: ...
