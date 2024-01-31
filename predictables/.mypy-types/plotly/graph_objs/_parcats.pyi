from plotly.basedatatypes import BaseTraceType as _BaseTraceType
from typing import Any

class Parcats(_BaseTraceType):
    @property
    def arrangement(self): ...
    @arrangement.setter
    def arrangement(self, val) -> None: ...
    @property
    def bundlecolors(self): ...
    @bundlecolors.setter
    def bundlecolors(self, val) -> None: ...
    @property
    def counts(self): ...
    @counts.setter
    def counts(self, val) -> None: ...
    @property
    def countssrc(self): ...
    @countssrc.setter
    def countssrc(self, val) -> None: ...
    @property
    def dimensions(self): ...
    @dimensions.setter
    def dimensions(self, val) -> None: ...
    @property
    def dimensiondefaults(self): ...
    @dimensiondefaults.setter
    def dimensiondefaults(self, val) -> None: ...
    @property
    def domain(self): ...
    @domain.setter
    def domain(self, val) -> None: ...
    @property
    def hoverinfo(self): ...
    @hoverinfo.setter
    def hoverinfo(self, val) -> None: ...
    @property
    def hoveron(self): ...
    @hoveron.setter
    def hoveron(self, val) -> None: ...
    @property
    def hovertemplate(self): ...
    @hovertemplate.setter
    def hovertemplate(self, val) -> None: ...
    @property
    def labelfont(self): ...
    @labelfont.setter
    def labelfont(self, val) -> None: ...
    @property
    def legendgrouptitle(self): ...
    @legendgrouptitle.setter
    def legendgrouptitle(self, val) -> None: ...
    @property
    def legendwidth(self): ...
    @legendwidth.setter
    def legendwidth(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def meta(self): ...
    @meta.setter
    def meta(self, val) -> None: ...
    @property
    def metasrc(self): ...
    @metasrc.setter
    def metasrc(self, val) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def sortpaths(self): ...
    @sortpaths.setter
    def sortpaths(self, val) -> None: ...
    @property
    def stream(self): ...
    @stream.setter
    def stream(self, val) -> None: ...
    @property
    def tickfont(self): ...
    @tickfont.setter
    def tickfont(self, val) -> None: ...
    @property
    def uid(self): ...
    @uid.setter
    def uid(self, val) -> None: ...
    @property
    def uirevision(self): ...
    @uirevision.setter
    def uirevision(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    @property
    def type(self): ...
    def __init__(
        self,
        arg: Any | None = ...,
        arrangement: Any | None = ...,
        bundlecolors: Any | None = ...,
        counts: Any | None = ...,
        countssrc: Any | None = ...,
        dimensions: Any | None = ...,
        dimensiondefaults: Any | None = ...,
        domain: Any | None = ...,
        hoverinfo: Any | None = ...,
        hoveron: Any | None = ...,
        hovertemplate: Any | None = ...,
        labelfont: Any | None = ...,
        legendgrouptitle: Any | None = ...,
        legendwidth: Any | None = ...,
        line: Any | None = ...,
        meta: Any | None = ...,
        metasrc: Any | None = ...,
        name: Any | None = ...,
        sortpaths: Any | None = ...,
        stream: Any | None = ...,
        tickfont: Any | None = ...,
        uid: Any | None = ...,
        uirevision: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...