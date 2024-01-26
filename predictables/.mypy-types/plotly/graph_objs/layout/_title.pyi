from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Title(_BaseLayoutHierarchyType):
    @property
    def automargin(self): ...
    @automargin.setter
    def automargin(self, val) -> None: ...
    @property
    def font(self): ...
    @font.setter
    def font(self, val) -> None: ...
    @property
    def pad(self): ...
    @pad.setter
    def pad(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    @property
    def x(self): ...
    @x.setter
    def x(self, val) -> None: ...
    @property
    def xanchor(self): ...
    @xanchor.setter
    def xanchor(self, val) -> None: ...
    @property
    def xref(self): ...
    @xref.setter
    def xref(self, val) -> None: ...
    @property
    def y(self): ...
    @y.setter
    def y(self, val) -> None: ...
    @property
    def yanchor(self): ...
    @yanchor.setter
    def yanchor(self, val) -> None: ...
    @property
    def yref(self): ...
    @yref.setter
    def yref(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        automargin: Any | None = ...,
        font: Any | None = ...,
        pad: Any | None = ...,
        text: Any | None = ...,
        x: Any | None = ...,
        xanchor: Any | None = ...,
        xref: Any | None = ...,
        y: Any | None = ...,
        yanchor: Any | None = ...,
        yref: Any | None = ...,
        **kwargs
    ) -> None: ...
