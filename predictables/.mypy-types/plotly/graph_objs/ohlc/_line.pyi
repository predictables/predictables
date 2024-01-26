from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Line(_BaseTraceHierarchyType):
    @property
    def dash(self): ...
    @dash.setter
    def dash(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        dash: Any | None = ...,
        width: Any | None = ...,
        **kwargs
    ) -> None: ...
