from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Transition(_BaseLayoutHierarchyType):
    @property
    def duration(self): ...
    @duration.setter
    def duration(self, val) -> None: ...
    @property
    def easing(self): ...
    @easing.setter
    def easing(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        duration: Any | None = ...,
        easing: Any | None = ...,
        **kwargs
    ) -> None: ...
