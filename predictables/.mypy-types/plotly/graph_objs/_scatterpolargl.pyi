from plotly.basedatatypes import BaseTraceType as _BaseTraceType
from typing import Any

class Scatterpolargl(_BaseTraceType):
    @property
    def connectgaps(self): ...
    @connectgaps.setter
    def connectgaps(self, val) -> None: ...
    @property
    def customdata(self): ...
    @customdata.setter
    def customdata(self, val) -> None: ...
    @property
    def customdatasrc(self): ...
    @customdatasrc.setter
    def customdatasrc(self, val) -> None: ...
    @property
    def dr(self): ...
    @dr.setter
    def dr(self, val) -> None: ...
    @property
    def dtheta(self): ...
    @dtheta.setter
    def dtheta(self, val) -> None: ...
    @property
    def fill(self): ...
    @fill.setter
    def fill(self, val) -> None: ...
    @property
    def fillcolor(self): ...
    @fillcolor.setter
    def fillcolor(self, val) -> None: ...
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
    def hovertemplate(self): ...
    @hovertemplate.setter
    def hovertemplate(self, val) -> None: ...
    @property
    def hovertemplatesrc(self): ...
    @hovertemplatesrc.setter
    def hovertemplatesrc(self, val) -> None: ...
    @property
    def hovertext(self): ...
    @hovertext.setter
    def hovertext(self, val) -> None: ...
    @property
    def hovertextsrc(self): ...
    @hovertextsrc.setter
    def hovertextsrc(self, val) -> None: ...
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
    def legendgroup(self): ...
    @legendgroup.setter
    def legendgroup(self, val) -> None: ...
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
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def marker(self): ...
    @marker.setter
    def marker(self, val) -> None: ...
    @property
    def meta(self): ...
    @meta.setter
    def meta(self, val) -> None: ...
    @property
    def metasrc(self): ...
    @metasrc.setter
    def metasrc(self, val) -> None: ...
    @property
    def mode(self): ...
    @mode.setter
    def mode(self, val) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def r(self): ...
    @r.setter
    def r(self, val) -> None: ...
    @property
    def r0(self): ...
    @r0.setter
    def r0(self, val) -> None: ...
    @property
    def rsrc(self): ...
    @rsrc.setter
    def rsrc(self, val) -> None: ...
    @property
    def selected(self): ...
    @selected.setter
    def selected(self, val) -> None: ...
    @property
    def selectedpoints(self): ...
    @selectedpoints.setter
    def selectedpoints(self, val) -> None: ...
    @property
    def showlegend(self): ...
    @showlegend.setter
    def showlegend(self, val) -> None: ...
    @property
    def stream(self): ...
    @stream.setter
    def stream(self, val) -> None: ...
    @property
    def subplot(self): ...
    @subplot.setter
    def subplot(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    @property
    def textfont(self): ...
    @textfont.setter
    def textfont(self, val) -> None: ...
    @property
    def textposition(self): ...
    @textposition.setter
    def textposition(self, val) -> None: ...
    @property
    def textpositionsrc(self): ...
    @textpositionsrc.setter
    def textpositionsrc(self, val) -> None: ...
    @property
    def textsrc(self): ...
    @textsrc.setter
    def textsrc(self, val) -> None: ...
    @property
    def texttemplate(self): ...
    @texttemplate.setter
    def texttemplate(self, val) -> None: ...
    @property
    def texttemplatesrc(self): ...
    @texttemplatesrc.setter
    def texttemplatesrc(self, val) -> None: ...
    @property
    def theta(self): ...
    @theta.setter
    def theta(self, val) -> None: ...
    @property
    def theta0(self): ...
    @theta0.setter
    def theta0(self, val) -> None: ...
    @property
    def thetasrc(self): ...
    @thetasrc.setter
    def thetasrc(self, val) -> None: ...
    @property
    def thetaunit(self): ...
    @thetaunit.setter
    def thetaunit(self, val) -> None: ...
    @property
    def uid(self): ...
    @uid.setter
    def uid(self, val) -> None: ...
    @property
    def uirevision(self): ...
    @uirevision.setter
    def uirevision(self, val) -> None: ...
    @property
    def unselected(self): ...
    @unselected.setter
    def unselected(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    @property
    def type(self): ...
    def __init__(
        self,
        arg: Any | None = ...,
        connectgaps: Any | None = ...,
        customdata: Any | None = ...,
        customdatasrc: Any | None = ...,
        dr: Any | None = ...,
        dtheta: Any | None = ...,
        fill: Any | None = ...,
        fillcolor: Any | None = ...,
        hoverinfo: Any | None = ...,
        hoverinfosrc: Any | None = ...,
        hoverlabel: Any | None = ...,
        hovertemplate: Any | None = ...,
        hovertemplatesrc: Any | None = ...,
        hovertext: Any | None = ...,
        hovertextsrc: Any | None = ...,
        ids: Any | None = ...,
        idssrc: Any | None = ...,
        legend: Any | None = ...,
        legendgroup: Any | None = ...,
        legendgrouptitle: Any | None = ...,
        legendrank: Any | None = ...,
        legendwidth: Any | None = ...,
        line: Any | None = ...,
        marker: Any | None = ...,
        meta: Any | None = ...,
        metasrc: Any | None = ...,
        mode: Any | None = ...,
        name: Any | None = ...,
        opacity: Any | None = ...,
        r: Any | None = ...,
        r0: Any | None = ...,
        rsrc: Any | None = ...,
        selected: Any | None = ...,
        selectedpoints: Any | None = ...,
        showlegend: Any | None = ...,
        stream: Any | None = ...,
        subplot: Any | None = ...,
        text: Any | None = ...,
        textfont: Any | None = ...,
        textposition: Any | None = ...,
        textpositionsrc: Any | None = ...,
        textsrc: Any | None = ...,
        texttemplate: Any | None = ...,
        texttemplatesrc: Any | None = ...,
        theta: Any | None = ...,
        theta0: Any | None = ...,
        thetasrc: Any | None = ...,
        thetaunit: Any | None = ...,
        uid: Any | None = ...,
        uirevision: Any | None = ...,
        unselected: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...