from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Axis(_BaseTraceHierarchyType):
    @property
    def matches(self): ...
    @matches.setter
    def matches(self, val) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        matches: Any | None = ...,
        type: Any | None = ...,
        **kwargs
    ) -> None: ...
