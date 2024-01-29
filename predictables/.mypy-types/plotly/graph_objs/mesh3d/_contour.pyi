from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Contour(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def show(self): ...
    @show.setter
    def show(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        show: Any | None = ...,
        width: Any | None = ...,
        **kwargs
    ) -> None: ...
