from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Line(_BaseTraceHierarchyType):
    @property
    def autocolorscale(self): ...
    @autocolorscale.setter
    def autocolorscale(self, val) -> None: ...
    @property
    def cauto(self): ...
    @cauto.setter
    def cauto(self, val) -> None: ...
    @property
    def cmax(self): ...
    @cmax.setter
    def cmax(self, val) -> None: ...
    @property
    def cmid(self): ...
    @cmid.setter
    def cmid(self, val) -> None: ...
    @property
    def cmin(self): ...
    @cmin.setter
    def cmin(self, val) -> None: ...
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def coloraxis(self): ...
    @coloraxis.setter
    def coloraxis(self, val) -> None: ...
    @property
    def colorscale(self): ...
    @colorscale.setter
    def colorscale(self, val) -> None: ...
    @property
    def colorsrc(self): ...
    @colorsrc.setter
    def colorsrc(self, val) -> None: ...
    @property
    def reversescale(self): ...
    @reversescale.setter
    def reversescale(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    @property
    def widthsrc(self): ...
    @widthsrc.setter
    def widthsrc(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        autocolorscale: Any | None = ...,
        cauto: Any | None = ...,
        cmax: Any | None = ...,
        cmid: Any | None = ...,
        cmin: Any | None = ...,
        color: Any | None = ...,
        coloraxis: Any | None = ...,
        colorscale: Any | None = ...,
        colorsrc: Any | None = ...,
        reversescale: Any | None = ...,
        width: Any | None = ...,
        widthsrc: Any | None = ...,
        **kwargs
    ) -> None: ...
