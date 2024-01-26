from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Line(_BaseLayoutHierarchyType):
    @property
    def dash(self): ...
    @dash.setter
    def dash(self, val) -> None: ...
    @property
    def dashsrc(self): ...
    @dashsrc.setter
    def dashsrc(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        dash: Any | None = ...,
        dashsrc: Any | None = ...,
        width: Any | None = ...,
        **kwargs
    ) -> None: ...
