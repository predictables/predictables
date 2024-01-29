from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Title(_BaseTraceHierarchyType):
    @property
    def font(self): ...
    @font.setter
    def font(self, val) -> None: ...
    @property
    def side(self): ...
    @side.setter
    def side(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        font: Any | None = ...,
        side: Any | None = ...,
        text: Any | None = ...,
        **kwargs
    ) -> None: ...
