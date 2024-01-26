from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Starts(_BaseTraceHierarchyType):
    @property
    def x(self): ...
    @x.setter
    def x(self, val) -> None: ...
    @property
    def xsrc(self): ...
    @xsrc.setter
    def xsrc(self, val) -> None: ...
    @property
    def y(self): ...
    @y.setter
    def y(self, val) -> None: ...
    @property
    def ysrc(self): ...
    @ysrc.setter
    def ysrc(self, val) -> None: ...
    @property
    def z(self): ...
    @z.setter
    def z(self, val) -> None: ...
    @property
    def zsrc(self): ...
    @zsrc.setter
    def zsrc(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        x: Any | None = ...,
        xsrc: Any | None = ...,
        y: Any | None = ...,
        ysrc: Any | None = ...,
        z: Any | None = ...,
        zsrc: Any | None = ...,
        **kwargs
    ) -> None: ...
