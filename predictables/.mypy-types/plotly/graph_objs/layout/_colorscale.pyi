from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Colorscale(_BaseLayoutHierarchyType):
    @property
    def diverging(self): ...
    @diverging.setter
    def diverging(self, val) -> None: ...
    @property
    def sequential(self): ...
    @sequential.setter
    def sequential(self, val) -> None: ...
    @property
    def sequentialminus(self): ...
    @sequentialminus.setter
    def sequentialminus(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        diverging: Any | None = ...,
        sequential: Any | None = ...,
        sequentialminus: Any | None = ...,
        **kwargs
    ) -> None: ...
