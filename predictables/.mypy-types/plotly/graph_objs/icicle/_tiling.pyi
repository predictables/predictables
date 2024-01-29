from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Tiling(_BaseTraceHierarchyType):
    @property
    def flip(self): ...
    @flip.setter
    def flip(self, val) -> None: ...
    @property
    def orientation(self): ...
    @orientation.setter
    def orientation(self, val) -> None: ...
    @property
    def pad(self): ...
    @pad.setter
    def pad(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        flip: Any | None = ...,
        orientation: Any | None = ...,
        pad: Any | None = ...,
        **kwargs
    ) -> None: ...
