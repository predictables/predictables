from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Line(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        opacity: Any | None = ...,
        **kwargs
    ) -> None: ...
