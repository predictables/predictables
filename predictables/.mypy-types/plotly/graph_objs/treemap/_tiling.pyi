from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Tiling(_BaseTraceHierarchyType):
    @property
    def flip(self): ...
    @flip.setter
    def flip(self, val) -> None: ...
    @property
    def packing(self): ...
    @packing.setter
    def packing(self, val) -> None: ...
    @property
    def pad(self): ...
    @pad.setter
    def pad(self, val) -> None: ...
    @property
    def squarifyratio(self): ...
    @squarifyratio.setter
    def squarifyratio(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        flip: Any | None = ...,
        packing: Any | None = ...,
        pad: Any | None = ...,
        squarifyratio: Any | None = ...,
        **kwargs
    ) -> None: ...