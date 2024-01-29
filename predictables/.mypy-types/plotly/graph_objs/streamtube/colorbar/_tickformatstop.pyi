from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Tickformatstop(_BaseTraceHierarchyType):
    @property
    def dtickrange(self): ...
    @dtickrange.setter
    def dtickrange(self, val) -> None: ...
    @property
    def enabled(self): ...
    @enabled.setter
    def enabled(self, val) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def templateitemname(self): ...
    @templateitemname.setter
    def templateitemname(self, val) -> None: ...
    @property
    def value(self): ...
    @value.setter
    def value(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        dtickrange: Any | None = ...,
        enabled: Any | None = ...,
        name: Any | None = ...,
        templateitemname: Any | None = ...,
        value: Any | None = ...,
        **kwargs
    ) -> None: ...
