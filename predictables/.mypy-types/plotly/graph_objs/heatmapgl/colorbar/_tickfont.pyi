from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Tickfont(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def family(self): ...
    @family.setter
    def family(self, val) -> None: ...
    @property
    def size(self): ...
    @size.setter
    def size(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        family: Any | None = ...,
        size: Any | None = ...,
        **kwargs
    ) -> None: ...
