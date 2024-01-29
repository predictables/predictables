from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Fill(_BaseLayoutHierarchyType):
    @property
    def outlinecolor(self): ...
    @outlinecolor.setter
    def outlinecolor(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., outlinecolor: Any | None = ..., **kwargs
    ) -> None: ...
