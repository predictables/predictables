from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Textfont(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., color: Any | None = ..., **kwargs
    ) -> None: ...
