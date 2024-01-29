from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Cumulative(_BaseTraceHierarchyType):
    @property
    def currentbin(self): ...
    @currentbin.setter
    def currentbin(self, val) -> None: ...
    @property
    def direction(self): ...
    @direction.setter
    def direction(self, val) -> None: ...
    @property
    def enabled(self): ...
    @enabled.setter
    def enabled(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        currentbin: Any | None = ...,
        direction: Any | None = ...,
        enabled: Any | None = ...,
        **kwargs
    ) -> None: ...
