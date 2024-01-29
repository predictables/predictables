from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Hoverlabel(_BaseLayoutHierarchyType):
    @property
    def bgcolor(self): ...
    @bgcolor.setter
    def bgcolor(self, val) -> None: ...
    @property
    def bordercolor(self): ...
    @bordercolor.setter
    def bordercolor(self, val) -> None: ...
    @property
    def font(self): ...
    @font.setter
    def font(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        bgcolor: Any | None = ...,
        bordercolor: Any | None = ...,
        font: Any | None = ...,
        **kwargs
    ) -> None: ...
