from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Scene(_BaseLayoutHierarchyType):
    @property
    def annotations(self): ...
    @annotations.setter
    def annotations(self, val) -> None: ...
    @property
    def annotationdefaults(self): ...
    @annotationdefaults.setter
    def annotationdefaults(self, val) -> None: ...
    @property
    def aspectmode(self): ...
    @aspectmode.setter
    def aspectmode(self, val) -> None: ...
    @property
    def aspectratio(self): ...
    @aspectratio.setter
    def aspectratio(self, val) -> None: ...
    @property
    def bgcolor(self): ...
    @bgcolor.setter
    def bgcolor(self, val) -> None: ...
    @property
    def camera(self): ...
    @camera.setter
    def camera(self, val) -> None: ...
    @property
    def domain(self): ...
    @domain.setter
    def domain(self, val) -> None: ...
    @property
    def dragmode(self): ...
    @dragmode.setter
    def dragmode(self, val) -> None: ...
    @property
    def hovermode(self): ...
    @hovermode.setter
    def hovermode(self, val) -> None: ...
    @property
    def uirevision(self): ...
    @uirevision.setter
    def uirevision(self, val) -> None: ...
    @property
    def xaxis(self): ...
    @xaxis.setter
    def xaxis(self, val) -> None: ...
    @property
    def yaxis(self): ...
    @yaxis.setter
    def yaxis(self, val) -> None: ...
    @property
    def zaxis(self): ...
    @zaxis.setter
    def zaxis(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        annotations: Any | None = ...,
        annotationdefaults: Any | None = ...,
        aspectmode: Any | None = ...,
        aspectratio: Any | None = ...,
        bgcolor: Any | None = ...,
        camera: Any | None = ...,
        domain: Any | None = ...,
        dragmode: Any | None = ...,
        hovermode: Any | None = ...,
        uirevision: Any | None = ...,
        xaxis: Any | None = ...,
        yaxis: Any | None = ...,
        zaxis: Any | None = ...,
        **kwargs
    ) -> None: ...