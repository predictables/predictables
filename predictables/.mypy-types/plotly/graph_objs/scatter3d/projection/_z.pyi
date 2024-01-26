from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Z(_BaseTraceHierarchyType):
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def scale(self): ...
    @scale.setter
    def scale(self, val) -> None: ...
    @property
    def show(self): ...
    @show.setter
    def show(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        opacity: Any | None = ...,
        scale: Any | None = ...,
        show: Any | None = ...,
        **kwargs
    ) -> None: ...
