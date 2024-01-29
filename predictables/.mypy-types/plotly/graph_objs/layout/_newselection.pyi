from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Newselection(_BaseLayoutHierarchyType):
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def mode(self): ...
    @mode.setter
    def mode(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        line: Any | None = ...,
        mode: Any | None = ...,
        **kwargs
    ) -> None: ...
