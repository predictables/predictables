from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Shape(_BaseLayoutHierarchyType):
    @property
    def editable(self): ...
    @editable.setter
    def editable(self, val) -> None: ...
    @property
    def fillcolor(self): ...
    @fillcolor.setter
    def fillcolor(self, val) -> None: ...
    @property
    def fillrule(self): ...
    @fillrule.setter
    def fillrule(self, val) -> None: ...
    @property
    def label(self): ...
    @label.setter
    def label(self, val) -> None: ...
    @property
    def layer(self): ...
    @layer.setter
    def layer(self, val) -> None: ...
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
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def path(self): ...
    @path.setter
    def path(self, val) -> None: ...
    @property
    def showlegend(self): ...
    @showlegend.setter
    def showlegend(self, val) -> None: ...
    @property
    def templateitemname(self): ...
    @templateitemname.setter
    def templateitemname(self, val) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    @property
    def x0(self): ...
    @x0.setter
    def x0(self, val) -> None: ...
    @property
    def x1(self): ...
    @x1.setter
    def x1(self, val) -> None: ...
    @property
    def xanchor(self): ...
    @xanchor.setter
    def xanchor(self, val) -> None: ...
    @property
    def xref(self): ...
    @xref.setter
    def xref(self, val) -> None: ...
    @property
    def xsizemode(self): ...
    @xsizemode.setter
    def xsizemode(self, val) -> None: ...
    @property
    def y0(self): ...
    @y0.setter
    def y0(self, val) -> None: ...
    @property
    def y1(self): ...
    @y1.setter
    def y1(self, val) -> None: ...
    @property
    def yanchor(self): ...
    @yanchor.setter
    def yanchor(self, val) -> None: ...
    @property
    def yref(self): ...
    @yref.setter
    def yref(self, val) -> None: ...
    @property
    def ysizemode(self): ...
    @ysizemode.setter
    def ysizemode(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        editable: Any | None = ...,
        fillcolor: Any | None = ...,
        fillrule: Any | None = ...,
        label: Any | None = ...,
        layer: Any | None = ...,
        legend: Any | None = ...,
        legendgroup: Any | None = ...,
        legendgrouptitle: Any | None = ...,
        legendrank: Any | None = ...,
        legendwidth: Any | None = ...,
        line: Any | None = ...,
        name: Any | None = ...,
        opacity: Any | None = ...,
        path: Any | None = ...,
        showlegend: Any | None = ...,
        templateitemname: Any | None = ...,
        type: Any | None = ...,
        visible: Any | None = ...,
        x0: Any | None = ...,
        x1: Any | None = ...,
        xanchor: Any | None = ...,
        xref: Any | None = ...,
        xsizemode: Any | None = ...,
        y0: Any | None = ...,
        y1: Any | None = ...,
        yanchor: Any | None = ...,
        yref: Any | None = ...,
        ysizemode: Any | None = ...,
        **kwargs
    ) -> None: ...