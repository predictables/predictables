from plotly.basedatatypes import BaseTraceType as _BaseTraceType
from typing import Any

class Box(_BaseTraceType):
    @property
    def alignmentgroup(self): ...
    @alignmentgroup.setter
    def alignmentgroup(self, val) -> None: ...
    @property
    def boxmean(self): ...
    @boxmean.setter
    def boxmean(self, val) -> None: ...
    @property
    def boxpoints(self): ...
    @boxpoints.setter
    def boxpoints(self, val) -> None: ...
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
    def hoveron(self): ...
    @hoveron.setter
    def hoveron(self, val) -> None: ...
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
    def jitter(self): ...
    @jitter.setter
    def jitter(self, val) -> None: ...
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
    def lowerfence(self): ...
    @lowerfence.setter
    def lowerfence(self, val) -> None: ...
    @property
    def lowerfencesrc(self): ...
    @lowerfencesrc.setter
    def lowerfencesrc(self, val) -> None: ...
    @property
    def marker(self): ...
    @marker.setter
    def marker(self, val) -> None: ...
    @property
    def mean(self): ...
    @mean.setter
    def mean(self, val) -> None: ...
    @property
    def meansrc(self): ...
    @meansrc.setter
    def meansrc(self, val) -> None: ...
    @property
    def median(self): ...
    @median.setter
    def median(self, val) -> None: ...
    @property
    def mediansrc(self): ...
    @mediansrc.setter
    def mediansrc(self, val) -> None: ...
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
    def notched(self): ...
    @notched.setter
    def notched(self, val) -> None: ...
    @property
    def notchspan(self): ...
    @notchspan.setter
    def notchspan(self, val) -> None: ...
    @property
    def notchspansrc(self): ...
    @notchspansrc.setter
    def notchspansrc(self, val) -> None: ...
    @property
    def notchwidth(self): ...
    @notchwidth.setter
    def notchwidth(self, val) -> None: ...
    @property
    def offsetgroup(self): ...
    @offsetgroup.setter
    def offsetgroup(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def orientation(self): ...
    @orientation.setter
    def orientation(self, val) -> None: ...
    @property
    def pointpos(self): ...
    @pointpos.setter
    def pointpos(self, val) -> None: ...
    @property
    def q1(self): ...
    @q1.setter
    def q1(self, val) -> None: ...
    @property
    def q1src(self): ...
    @q1src.setter
    def q1src(self, val) -> None: ...
    @property
    def q3(self): ...
    @q3.setter
    def q3(self, val) -> None: ...
    @property
    def q3src(self): ...
    @q3src.setter
    def q3src(self, val) -> None: ...
    @property
    def quartilemethod(self): ...
    @quartilemethod.setter
    def quartilemethod(self, val) -> None: ...
    @property
    def sd(self): ...
    @sd.setter
    def sd(self, val) -> None: ...
    @property
    def sdmultiple(self): ...
    @sdmultiple.setter
    def sdmultiple(self, val) -> None: ...
    @property
    def sdsrc(self): ...
    @sdsrc.setter
    def sdsrc(self, val) -> None: ...
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
    def showwhiskers(self): ...
    @showwhiskers.setter
    def showwhiskers(self, val) -> None: ...
    @property
    def sizemode(self): ...
    @sizemode.setter
    def sizemode(self, val) -> None: ...
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
    def upperfence(self): ...
    @upperfence.setter
    def upperfence(self, val) -> None: ...
    @property
    def upperfencesrc(self): ...
    @upperfencesrc.setter
    def upperfencesrc(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    @property
    def whiskerwidth(self): ...
    @whiskerwidth.setter
    def whiskerwidth(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
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
    def xcalendar(self): ...
    @xcalendar.setter
    def xcalendar(self, val) -> None: ...
    @property
    def xhoverformat(self): ...
    @xhoverformat.setter
    def xhoverformat(self, val) -> None: ...
    @property
    def xperiod(self): ...
    @xperiod.setter
    def xperiod(self, val) -> None: ...
    @property
    def xperiod0(self): ...
    @xperiod0.setter
    def xperiod0(self, val) -> None: ...
    @property
    def xperiodalignment(self): ...
    @xperiodalignment.setter
    def xperiodalignment(self, val) -> None: ...
    @property
    def xsrc(self): ...
    @xsrc.setter
    def xsrc(self, val) -> None: ...
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
    def ycalendar(self): ...
    @ycalendar.setter
    def ycalendar(self, val) -> None: ...
    @property
    def yhoverformat(self): ...
    @yhoverformat.setter
    def yhoverformat(self, val) -> None: ...
    @property
    def yperiod(self): ...
    @yperiod.setter
    def yperiod(self, val) -> None: ...
    @property
    def yperiod0(self): ...
    @yperiod0.setter
    def yperiod0(self, val) -> None: ...
    @property
    def yperiodalignment(self): ...
    @yperiodalignment.setter
    def yperiodalignment(self, val) -> None: ...
    @property
    def ysrc(self): ...
    @ysrc.setter
    def ysrc(self, val) -> None: ...
    @property
    def type(self): ...
    def __init__(
        self,
        arg: Any | None = ...,
        alignmentgroup: Any | None = ...,
        boxmean: Any | None = ...,
        boxpoints: Any | None = ...,
        customdata: Any | None = ...,
        customdatasrc: Any | None = ...,
        dx: Any | None = ...,
        dy: Any | None = ...,
        fillcolor: Any | None = ...,
        hoverinfo: Any | None = ...,
        hoverinfosrc: Any | None = ...,
        hoverlabel: Any | None = ...,
        hoveron: Any | None = ...,
        hovertemplate: Any | None = ...,
        hovertemplatesrc: Any | None = ...,
        hovertext: Any | None = ...,
        hovertextsrc: Any | None = ...,
        ids: Any | None = ...,
        idssrc: Any | None = ...,
        jitter: Any | None = ...,
        legend: Any | None = ...,
        legendgroup: Any | None = ...,
        legendgrouptitle: Any | None = ...,
        legendrank: Any | None = ...,
        legendwidth: Any | None = ...,
        line: Any | None = ...,
        lowerfence: Any | None = ...,
        lowerfencesrc: Any | None = ...,
        marker: Any | None = ...,
        mean: Any | None = ...,
        meansrc: Any | None = ...,
        median: Any | None = ...,
        mediansrc: Any | None = ...,
        meta: Any | None = ...,
        metasrc: Any | None = ...,
        name: Any | None = ...,
        notched: Any | None = ...,
        notchspan: Any | None = ...,
        notchspansrc: Any | None = ...,
        notchwidth: Any | None = ...,
        offsetgroup: Any | None = ...,
        opacity: Any | None = ...,
        orientation: Any | None = ...,
        pointpos: Any | None = ...,
        q1: Any | None = ...,
        q1src: Any | None = ...,
        q3: Any | None = ...,
        q3src: Any | None = ...,
        quartilemethod: Any | None = ...,
        sd: Any | None = ...,
        sdmultiple: Any | None = ...,
        sdsrc: Any | None = ...,
        selected: Any | None = ...,
        selectedpoints: Any | None = ...,
        showlegend: Any | None = ...,
        showwhiskers: Any | None = ...,
        sizemode: Any | None = ...,
        stream: Any | None = ...,
        text: Any | None = ...,
        textsrc: Any | None = ...,
        uid: Any | None = ...,
        uirevision: Any | None = ...,
        unselected: Any | None = ...,
        upperfence: Any | None = ...,
        upperfencesrc: Any | None = ...,
        visible: Any | None = ...,
        whiskerwidth: Any | None = ...,
        width: Any | None = ...,
        x: Any | None = ...,
        x0: Any | None = ...,
        xaxis: Any | None = ...,
        xcalendar: Any | None = ...,
        xhoverformat: Any | None = ...,
        xperiod: Any | None = ...,
        xperiod0: Any | None = ...,
        xperiodalignment: Any | None = ...,
        xsrc: Any | None = ...,
        y: Any | None = ...,
        y0: Any | None = ...,
        yaxis: Any | None = ...,
        ycalendar: Any | None = ...,
        yhoverformat: Any | None = ...,
        yperiod: Any | None = ...,
        yperiod0: Any | None = ...,
        yperiodalignment: Any | None = ...,
        ysrc: Any | None = ...,
        **kwargs
    ) -> None: ...
