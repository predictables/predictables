from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Label(_BaseLayoutHierarchyType):
    @property
    def font(self): ...
    @font.setter
    def font(self, val) -> None: ...
    @property
    def padding(self): ...
    @padding.setter
    def padding(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    @property
    def textangle(self): ...
    @textangle.setter
    def textangle(self, val) -> None: ...
    @property
    def textposition(self): ...
    @textposition.setter
    def textposition(self, val) -> None: ...
    @property
    def texttemplate(self): ...
    @texttemplate.setter
    def texttemplate(self, val) -> None: ...
    @property
    def xanchor(self): ...
    @xanchor.setter
    def xanchor(self, val) -> None: ...
    @property
    def yanchor(self): ...
    @yanchor.setter
    def yanchor(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        font: Any | None = ...,
        padding: Any | None = ...,
        text: Any | None = ...,
        textangle: Any | None = ...,
        textposition: Any | None = ...,
        texttemplate: Any | None = ...,
        xanchor: Any | None = ...,
        yanchor: Any | None = ...,
        **kwargs
    ) -> None: ...