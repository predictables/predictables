from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Threshold(_BaseTraceHierarchyType):
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def thickness(self): ...
    @thickness.setter
    def thickness(self, val) -> None: ...
    @property
    def value(self): ...
    @value.setter
    def value(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        line: Any | None = ...,
        thickness: Any | None = ...,
        value: Any | None = ...,
        **kwargs
    ) -> None: ...
