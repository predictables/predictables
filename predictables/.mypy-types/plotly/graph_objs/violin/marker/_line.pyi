from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Line(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def outliercolor(self): ...
    @outliercolor.setter
    def outliercolor(self, val) -> None: ...
    @property
    def outlierwidth(self): ...
    @outlierwidth.setter
    def outlierwidth(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        outliercolor: Any | None = ...,
        outlierwidth: Any | None = ...,
        width: Any | None = ...,
        **kwargs
    ) -> None: ...
