from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class XBins(_BaseTraceHierarchyType):
    @property
    def end(self): ...
    @end.setter
    def end(self, val) -> None: ...
    @property
    def size(self): ...
    @size.setter
    def size(self, val) -> None: ...
    @property
    def start(self): ...
    @start.setter
    def start(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        end: Any | None = ...,
        size: Any | None = ...,
        start: Any | None = ...,
        **kwargs
    ) -> None: ...
