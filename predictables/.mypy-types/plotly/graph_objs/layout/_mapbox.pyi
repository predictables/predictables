from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Mapbox(_BaseLayoutHierarchyType):
    @property
    def accesstoken(self): ...
    @accesstoken.setter
    def accesstoken(self, val) -> None: ...
    @property
    def bearing(self): ...
    @bearing.setter
    def bearing(self, val) -> None: ...
    @property
    def bounds(self): ...
    @bounds.setter
    def bounds(self, val) -> None: ...
    @property
    def center(self): ...
    @center.setter
    def center(self, val) -> None: ...
    @property
    def domain(self): ...
    @domain.setter
    def domain(self, val) -> None: ...
    @property
    def layers(self): ...
    @layers.setter
    def layers(self, val) -> None: ...
    @property
    def layerdefaults(self): ...
    @layerdefaults.setter
    def layerdefaults(self, val) -> None: ...
    @property
    def pitch(self): ...
    @pitch.setter
    def pitch(self, val) -> None: ...
    @property
    def style(self): ...
    @style.setter
    def style(self, val) -> None: ...
    @property
    def uirevision(self): ...
    @uirevision.setter
    def uirevision(self, val) -> None: ...
    @property
    def zoom(self): ...
    @zoom.setter
    def zoom(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        accesstoken: Any | None = ...,
        bearing: Any | None = ...,
        bounds: Any | None = ...,
        center: Any | None = ...,
        domain: Any | None = ...,
        layers: Any | None = ...,
        layerdefaults: Any | None = ...,
        pitch: Any | None = ...,
        style: Any | None = ...,
        uirevision: Any | None = ...,
        zoom: Any | None = ...,
        **kwargs
    ) -> None: ...