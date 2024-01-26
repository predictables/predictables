from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Title(_BaseTraceHierarchyType):
    @property
    def font(self): ...
    @font.setter
    def font(self, val) -> None: ...
    @property
    def position(self): ...
    @position.setter
    def position(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        font: Any | None = ...,
        position: Any | None = ...,
        text: Any | None = ...,
        **kwargs
    ) -> None: ...
