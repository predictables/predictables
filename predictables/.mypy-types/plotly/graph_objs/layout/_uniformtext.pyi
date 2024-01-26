from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Uniformtext(_BaseLayoutHierarchyType):
    @property
    def minsize(self): ...
    @minsize.setter
    def minsize(self, val) -> None: ...
    @property
    def mode(self): ...
    @mode.setter
    def mode(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        minsize: Any | None = ...,
        mode: Any | None = ...,
        **kwargs
    ) -> None: ...
