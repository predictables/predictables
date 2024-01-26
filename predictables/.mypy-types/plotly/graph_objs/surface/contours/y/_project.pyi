from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Project(_BaseTraceHierarchyType):
    @property
    def x(self): ...
    @x.setter
    def x(self, val) -> None: ...
    @property
    def y(self): ...
    @y.setter
    def y(self, val) -> None: ...
    @property
    def z(self): ...
    @z.setter
    def z(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        x: Any | None = ...,
        y: Any | None = ...,
        z: Any | None = ...,
        **kwargs
    ) -> None: ...
