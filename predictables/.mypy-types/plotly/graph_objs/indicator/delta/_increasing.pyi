from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Increasing(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def symbol(self): ...
    @symbol.setter
    def symbol(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        symbol: Any | None = ...,
        **kwargs
    ) -> None: ...
