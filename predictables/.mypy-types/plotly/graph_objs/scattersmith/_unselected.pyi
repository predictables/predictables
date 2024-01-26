from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Unselected(_BaseTraceHierarchyType):
    @property
    def marker(self): ...
    @marker.setter
    def marker(self, val) -> None: ...
    @property
    def textfont(self): ...
    @textfont.setter
    def textfont(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        marker: Any | None = ...,
        textfont: Any | None = ...,
        **kwargs
    ) -> None: ...
