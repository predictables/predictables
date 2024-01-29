from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Template(_BaseLayoutHierarchyType):
    @property
    def data(self): ...
    @data.setter
    def data(self, val) -> None: ...
    @property
    def layout(self): ...
    @layout.setter
    def layout(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        data: Any | None = ...,
        layout: Any | None = ...,
        **kwargs
    ) -> None: ...
