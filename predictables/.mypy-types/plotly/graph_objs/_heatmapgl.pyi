from plotly.basedatatypes import BaseTraceType as _BaseTraceType
from typing import Any

class Heatmapgl(_BaseTraceType):
    @property
    def autocolorscale(self): ...
    @autocolorscale.setter
    def autocolorscale(self, val) -> None: ...
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
    def customdata(self): ...
    @customdata.setter
    def customdata(self, val) -> None: ...
    @property
    def customdatasrc(self): ...
    @customdatasrc.setter
    def customdatasrc(self, val) -> None: ...
    @property
    def dx(self): ...
    @dx.setter
    def dx(self, val) -> None: ...
    @property
    def dy(self): ...
    @dy.setter
    def dy(self, val) -> None: ...
    @property
    def hoverinfo(self): ...
    @hoverinfo.setter
    def hoverinfo(self, val) -> None: ...
    @property
    def hoverinfosrc(self): ...
    @hoverinfosrc.setter
    def hoverinfosrc(self, val) -> None: ...
    @property
    def hoverlabel(self): ...
    @hoverlabel.setter
    def hoverlabel(self, val) -> None: ...
    @property
    def ids(self): ...
    @ids.setter
    def ids(self, val) -> None: ...
    @property
    def idssrc(self): ...
    @idssrc.setter
    def idssrc(self, val) -> None: ...
    @property
    def legend(self): ...
    @legend.setter
    def legend(self, val) -> None: ...
    @property
    def legendgrouptitle(self): ...
    @legendgrouptitle.setter
    def legendgrouptitle(self, val) -> None: ...
    @property
    def legendrank(self): ...
    @legendrank.setter
    def legendrank(self, val) -> None: ...
    @property
    def legendwidth(self): ...
    @legendwidth.setter
    def legendwidth(self, val) -> None: ...
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
    def stream(self): ...
    @stream.setter
    def stream(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    @property
    def textsrc(self): ...
    @textsrc.setter
    def textsrc(self, val) -> None: ...
    @property
    def transpose(self): ...
    @transpose.setter
    def transpose(self, val) -> None: ...
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
    def x(self): ...
    @x.setter
    def x(self, val) -> None: ...
    @property
    def x0(self): ...
    @x0.setter
    def x0(self, val) -> None: ...
    @property
    def xaxis(self): ...
    @xaxis.setter
    def xaxis(self, val) -> None: ...
    @property
    def xsrc(self): ...
    @xsrc.setter
    def xsrc(self, val) -> None: ...
    @property
    def xtype(self): ...
    @xtype.setter
    def xtype(self, val) -> None: ...
    @property
    def y(self): ...
    @y.setter
    def y(self, val) -> None: ...
    @property
    def y0(self): ...
    @y0.setter
    def y0(self, val) -> None: ...
    @property
    def yaxis(self): ...
    @yaxis.setter
    def yaxis(self, val) -> None: ...
    @property
    def ysrc(self): ...
    @ysrc.setter
    def ysrc(self, val) -> None: ...
    @property
    def ytype(self): ...
    @ytype.setter
    def ytype(self, val) -> None: ...
    @property
    def z(self): ...
    @z.setter
    def z(self, val) -> None: ...
    @property
    def zauto(self): ...
    @zauto.setter
    def zauto(self, val) -> None: ...
    @property
    def zmax(self): ...
    @zmax.setter
    def zmax(self, val) -> None: ...
    @property
    def zmid(self): ...
    @zmid.setter
    def zmid(self, val) -> None: ...
    @property
    def zmin(self): ...
    @zmin.setter
    def zmin(self, val) -> None: ...
    @property
    def zsmooth(self): ...
    @zsmooth.setter
    def zsmooth(self, val) -> None: ...
    @property
    def zsrc(self): ...
    @zsrc.setter
    def zsrc(self, val) -> None: ...
    @property
    def type(self): ...
    def __init__(
        self,
        arg: Any | None = ...,
        autocolorscale: Any | None = ...,
        coloraxis: Any | None = ...,
        colorbar: Any | None = ...,
        colorscale: Any | None = ...,
        customdata: Any | None = ...,
        customdatasrc: Any | None = ...,
        dx: Any | None = ...,
        dy: Any | None = ...,
        hoverinfo: Any | None = ...,
        hoverinfosrc: Any | None = ...,
        hoverlabel: Any | None = ...,
        ids: Any | None = ...,
        idssrc: Any | None = ...,
        legend: Any | None = ...,
        legendgrouptitle: Any | None = ...,
        legendrank: Any | None = ...,
        legendwidth: Any | None = ...,
        meta: Any | None = ...,
        metasrc: Any | None = ...,
        name: Any | None = ...,
        opacity: Any | None = ...,
        reversescale: Any | None = ...,
        showscale: Any | None = ...,
        stream: Any | None = ...,
        text: Any | None = ...,
        textsrc: Any | None = ...,
        transpose: Any | None = ...,
        uid: Any | None = ...,
        uirevision: Any | None = ...,
        visible: Any | None = ...,
        x: Any | None = ...,
        x0: Any | None = ...,
        xaxis: Any | None = ...,
        xsrc: Any | None = ...,
        xtype: Any | None = ...,
        y: Any | None = ...,
        y0: Any | None = ...,
        yaxis: Any | None = ...,
        ysrc: Any | None = ...,
        ytype: Any | None = ...,
        z: Any | None = ...,
        zauto: Any | None = ...,
        zmax: Any | None = ...,
        zmid: Any | None = ...,
        zmin: Any | None = ...,
        zsmooth: Any | None = ...,
        zsrc: Any | None = ...,
        **kwargs
    ) -> None: ...