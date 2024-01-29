from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Projection(_BaseLayoutHierarchyType):
    @property
    def type(self): ...
    @type.setter
    def type(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., type: Any | None = ..., **kwargs
    ) -> None: ...
