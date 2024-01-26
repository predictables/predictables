from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Marker(_BaseTraceHierarchyType):
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
    def colorbar(self): ...
    @colorbar.setter
    def colorbar(self, val) -> None: ...
    @property
    def colorscale(self): ...
    @colorscale.setter
    def colorscale(self, val) -> None: ...
    @property
    def colorsrc(self): ...
    @colorsrc.setter
    def colorsrc(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def reversescale(self): ...
    @reversescale.setter
    def reversescale(self, val) -> None: ...
    @property
    def showscale(self): ...
    @showscale.setter
    def showscale(self, val) -> None: ...
    @property
    def size(self): ...
    @size.setter
    def size(self, val) -> None: ...
    @property
    def sizemin(self): ...
    @sizemin.setter
    def sizemin(self, val) -> None: ...
    @property
    def sizemode(self): ...
    @sizemode.setter
    def sizemode(self, val) -> None: ...
    @property
    def sizeref(self): ...
    @sizeref.setter
    def sizeref(self, val) -> None: ...
    @property
    def sizesrc(self): ...
    @sizesrc.setter
    def sizesrc(self, val) -> None: ...
    @property
    def symbol(self): ...
    @symbol.setter
    def symbol(self, val) -> None: ...
    @property
    def symbolsrc(self): ...
    @symbolsrc.setter
    def symbolsrc(self, val) -> None: ...
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
        colorbar: Any | None = ...,
        colorscale: Any | None = ...,
        colorsrc: Any | None = ...,
        line: Any | None = ...,
        opacity: Any | None = ...,
        reversescale: Any | None = ...,
        showscale: Any | None = ...,
        size: Any | None = ...,
        sizemin: Any | None = ...,
        sizemode: Any | None = ...,
        sizeref: Any | None = ...,
        sizesrc: Any | None = ...,
        symbol: Any | None = ...,
        symbolsrc: Any | None = ...,
        **kwargs
    ) -> None: ...
