from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Autorangeoptions(_BaseLayoutHierarchyType):
    @property
    def clipmax(self): ...
    @clipmax.setter
    def clipmax(self, val) -> None: ...
    @property
    def clipmin(self): ...
    @clipmin.setter
    def clipmin(self, val) -> None: ...
    @property
    def include(self): ...
    @include.setter
    def include(self, val) -> None: ...
    @property
    def includesrc(self): ...
    @includesrc.setter
    def includesrc(self, val) -> None: ...
    @property
    def maxallowed(self): ...
    @maxallowed.setter
    def maxallowed(self, val) -> None: ...
    @property
    def minallowed(self): ...
    @minallowed.setter
    def minallowed(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        clipmax: Any | None = ...,
        clipmin: Any | None = ...,
        include: Any | None = ...,
        includesrc: Any | None = ...,
        maxallowed: Any | None = ...,
        minallowed: Any | None = ...,
        **kwargs
    ) -> None: ...
