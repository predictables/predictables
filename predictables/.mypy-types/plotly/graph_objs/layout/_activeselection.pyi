from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Activeselection(_BaseLayoutHierarchyType):
    @property
    def fillcolor(self): ...
    @fillcolor.setter
    def fillcolor(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        fillcolor: Any | None = ...,
        opacity: Any | None = ...,
        **kwargs
    ) -> None: ...
