from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Connector(_BaseTraceHierarchyType):
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def mode(self): ...
    @mode.setter
    def mode(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        line: Any | None = ...,
        mode: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...
