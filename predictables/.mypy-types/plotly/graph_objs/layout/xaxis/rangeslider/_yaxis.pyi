from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class YAxis(_BaseLayoutHierarchyType):
    @property
    def range(self): ...
    @range.setter
    def range(self, val) -> None: ...
    @property
    def rangemode(self): ...
    @rangemode.setter
    def rangemode(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        range: Any | None = ...,
        rangemode: Any | None = ...,
        **kwargs
    ) -> None: ...
