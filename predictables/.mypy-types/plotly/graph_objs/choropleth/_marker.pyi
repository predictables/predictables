from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Marker(_BaseTraceHierarchyType):
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def opacitysrc(self): ...
    @opacitysrc.setter
    def opacitysrc(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        line: Any | None = ...,
        opacity: Any | None = ...,
        opacitysrc: Any | None = ...,
        **kwargs
    ) -> None: ...
