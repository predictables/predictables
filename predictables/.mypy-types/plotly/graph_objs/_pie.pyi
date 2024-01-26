from plotly.basedatatypes import BaseTraceType as _BaseTraceType
from typing import Any

class Pie(_BaseTraceType):
    @property
    def automargin(self): ...
    @automargin.setter
    def automargin(self, val) -> None: ...
    @property
    def customdata(self): ...
    @customdata.setter
    def customdata(self, val) -> None: ...
    @property
    def customdatasrc(self): ...
    @customdatasrc.setter
    def customdatasrc(self, val) -> None: ...
    @property
    def direction(self): ...
    @direction.setter
    def direction(self, val) -> None: ...
    @property
    def dlabel(self): ...
    @dlabel.setter
    def dlabel(self, val) -> None: ...
    @property
    def domain(self): ...
    @domain.setter
    def domain(self, val) -> None: ...
    @property
    def hole(self): ...
    @hole.setter
    def hole(self, val) -> None: ...
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
    def insidetextfont(self): ...
    @insidetextfont.setter
    def insidetextfont(self, val) -> None: ...
    @property
    def insidetextorientation(self): ...
    @insidetextorientation.setter
    def insidetextorientation(self, val) -> None: ...
    @property
    def label0(self): ...
    @label0.setter
    def label0(self, val) -> None: ...
    @property
    def labels(self): ...
    @labels.setter
    def labels(self, val) -> None: ...
    @property
    def labelssrc(self): ...
    @labelssrc.setter
    def labelssrc(self, val) -> None: ...
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
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def outsidetextfont(self): ...
    @outsidetextfont.setter
    def outsidetextfont(self, val) -> None: ...
    @property
    def pull(self): ...
    @pull.setter
    def pull(self, val) -> None: ...
    @property
    def pullsrc(self): ...
    @pullsrc.setter
    def pullsrc(self, val) -> None: ...
    @property
    def rotation(self): ...
    @rotation.setter
    def rotation(self, val) -> None: ...
    @property
    def scalegroup(self): ...
    @scalegroup.setter
    def scalegroup(self, val) -> None: ...
    @property
    def showlegend(self): ...
    @showlegend.setter
    def showlegend(self, val) -> None: ...
    @property
    def sort(self): ...
    @sort.setter
    def sort(self, val) -> None: ...
    @property
    def stream(self): ...
    @stream.setter
    def stream(self, val) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, val) -> None: ...
    @property
    def textfont(self): ...
    @textfont.setter
    def textfont(self, val) -> None: ...
    @property
    def textinfo(self): ...
    @textinfo.setter
    def textinfo(self, val) -> None: ...
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
    def title(self): ...
    @title.setter
    def title(self, val) -> None: ...
    @property
    def titlefont(self): ...
    @titlefont.setter
    def titlefont(self, val) -> None: ...
    @property
    def titleposition(self): ...
    @titleposition.setter
    def titleposition(self, val) -> None: ...
    @property
    def uid(self): ...
    @uid.setter
    def uid(self, val) -> None: ...
    @property
    def uirevision(self): ...
    @uirevision.setter
    def uirevision(self, val) -> None: ...
    @property
    def values(self): ...
    @values.setter
    def values(self, val) -> None: ...
    @property
    def valuessrc(self): ...
    @valuessrc.setter
    def valuessrc(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    @property
    def type(self): ...
    def __init__(
        self,
        arg: Any | None = ...,
        automargin: Any | None = ...,
        customdata: Any | None = ...,
        customdatasrc: Any | None = ...,
        direction: Any | None = ...,
        dlabel: Any | None = ...,
        domain: Any | None = ...,
        hole: Any | None = ...,
        hoverinfo: Any | None = ...,
        hoverinfosrc: Any | None = ...,
        hoverlabel: Any | None = ...,
        hovertemplate: Any | None = ...,
        hovertemplatesrc: Any | None = ...,
        hovertext: Any | None = ...,
        hovertextsrc: Any | None = ...,
        ids: Any | None = ...,
        idssrc: Any | None = ...,
        insidetextfont: Any | None = ...,
        insidetextorientation: Any | None = ...,
        label0: Any | None = ...,
        labels: Any | None = ...,
        labelssrc: Any | None = ...,
        legend: Any | None = ...,
        legendgroup: Any | None = ...,
        legendgrouptitle: Any | None = ...,
        legendrank: Any | None = ...,
        legendwidth: Any | None = ...,
        marker: Any | None = ...,
        meta: Any | None = ...,
        metasrc: Any | None = ...,
        name: Any | None = ...,
        opacity: Any | None = ...,
        outsidetextfont: Any | None = ...,
        pull: Any | None = ...,
        pullsrc: Any | None = ...,
        rotation: Any | None = ...,
        scalegroup: Any | None = ...,
        showlegend: Any | None = ...,
        sort: Any | None = ...,
        stream: Any | None = ...,
        text: Any | None = ...,
        textfont: Any | None = ...,
        textinfo: Any | None = ...,
        textposition: Any | None = ...,
        textpositionsrc: Any | None = ...,
        textsrc: Any | None = ...,
        texttemplate: Any | None = ...,
        texttemplatesrc: Any | None = ...,
        title: Any | None = ...,
        titlefont: Any | None = ...,
        titleposition: Any | None = ...,
        uid: Any | None = ...,
        uirevision: Any | None = ...,
        values: Any | None = ...,
        valuessrc: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...
