from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Border(_BaseTraceHierarchyType):
    @property
    def arearatio(self): ...
    @arearatio.setter
    def arearatio(self, val) -> None: ...
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        arearatio: Any | None = ...,
        color: Any | None = ...,
        **kwargs
    ) -> None: ...
