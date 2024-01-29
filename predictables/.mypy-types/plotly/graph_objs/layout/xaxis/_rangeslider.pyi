from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Rangeslider(_BaseLayoutHierarchyType):
    @property
    def autorange(self): ...
    @autorange.setter
    def autorange(self, val) -> None: ...
    @property
    def bgcolor(self): ...
    @bgcolor.setter
    def bgcolor(self, val) -> None: ...
    @property
    def bordercolor(self): ...
    @bordercolor.setter
    def bordercolor(self, val) -> None: ...
    @property
    def borderwidth(self): ...
    @borderwidth.setter
    def borderwidth(self, val) -> None: ...
    @property
    def range(self): ...
    @range.setter
    def range(self, val) -> None: ...
    @property
    def thickness(self): ...
    @thickness.setter
    def thickness(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    @property
    def yaxis(self): ...
    @yaxis.setter
    def yaxis(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        autorange: Any | None = ...,
        bgcolor: Any | None = ...,
        bordercolor: Any | None = ...,
        borderwidth: Any | None = ...,
        range: Any | None = ...,
        thickness: Any | None = ...,
        visible: Any | None = ...,
        yaxis: Any | None = ...,
        **kwargs
    ) -> None: ...
