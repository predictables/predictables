from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Circle(_BaseLayoutHierarchyType):
    @property
    def radius(self): ...
    @radius.setter
    def radius(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., radius: Any | None = ..., **kwargs
    ) -> None: ...
