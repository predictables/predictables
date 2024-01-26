from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Rotation(_BaseLayoutHierarchyType):
    @property
    def lat(self): ...
    @lat.setter
    def lat(self, val) -> None: ...
    @property
    def lon(self): ...
    @lon.setter
    def lon(self, val) -> None: ...
    @property
    def roll(self): ...
    @roll.setter
    def roll(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        lat: Any | None = ...,
        lon: Any | None = ...,
        roll: Any | None = ...,
        **kwargs
    ) -> None: ...
