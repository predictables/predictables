from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Spaceframe(_BaseTraceHierarchyType):
    @property
    def fill(self): ...
    @fill.setter
    def fill(self, val) -> None: ...
    @property
    def show(self): ...
    @show.setter
    def show(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        fill: Any | None = ...,
        show: Any | None = ...,
        **kwargs
    ) -> None: ...
