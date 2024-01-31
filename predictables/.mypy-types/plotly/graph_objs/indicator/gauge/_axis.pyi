from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Axis(_BaseTraceHierarchyType):
    @property
    def dtick(self): ...
    @dtick.setter
    def dtick(self, val) -> None: ...
    @property
    def exponentformat(self): ...
    @exponentformat.setter
    def exponentformat(self, val) -> None: ...
    @property
    def labelalias(self): ...
    @labelalias.setter
    def labelalias(self, val) -> None: ...
    @property
    def minexponent(self): ...
    @minexponent.setter
    def minexponent(self, val) -> None: ...
    @property
    def nticks(self): ...
    @nticks.setter
    def nticks(self, val) -> None: ...
    @property
    def range(self): ...
    @range.setter
    def range(self, val) -> None: ...
    @property
    def separatethousands(self): ...
    @separatethousands.setter
    def separatethousands(self, val) -> None: ...
    @property
    def showexponent(self): ...
    @showexponent.setter
    def showexponent(self, val) -> None: ...
    @property
    def showticklabels(self): ...
    @showticklabels.setter
    def showticklabels(self, val) -> None: ...
    @property
    def showtickprefix(self): ...
    @showtickprefix.setter
    def showtickprefix(self, val) -> None: ...
    @property
    def showticksuffix(self): ...
    @showticksuffix.setter
    def showticksuffix(self, val) -> None: ...
    @property
    def tick0(self): ...
    @tick0.setter
    def tick0(self, val) -> None: ...
    @property
    def tickangle(self): ...
    @tickangle.setter
    def tickangle(self, val) -> None: ...
    @property
    def tickcolor(self): ...
    @tickcolor.setter
    def tickcolor(self, val) -> None: ...
    @property
    def tickfont(self): ...
    @tickfont.setter
    def tickfont(self, val) -> None: ...
    @property
    def tickformat(self): ...
    @tickformat.setter
    def tickformat(self, val) -> None: ...
    @property
    def tickformatstops(self): ...
    @tickformatstops.setter
    def tickformatstops(self, val) -> None: ...
    @property
    def tickformatstopdefaults(self): ...
    @tickformatstopdefaults.setter
    def tickformatstopdefaults(self, val) -> None: ...
    @property
    def ticklabelstep(self): ...
    @ticklabelstep.setter
    def ticklabelstep(self, val) -> None: ...
    @property
    def ticklen(self): ...
    @ticklen.setter
    def ticklen(self, val) -> None: ...
    @property
    def tickmode(self): ...
    @tickmode.setter
    def tickmode(self, val) -> None: ...
    @property
    def tickprefix(self): ...
    @tickprefix.setter
    def tickprefix(self, val) -> None: ...
    @property
    def ticks(self): ...
    @ticks.setter
    def ticks(self, val) -> None: ...
    @property
    def ticksuffix(self): ...
    @ticksuffix.setter
    def ticksuffix(self, val) -> None: ...
    @property
    def ticktext(self): ...
    @ticktext.setter
    def ticktext(self, val) -> None: ...
    @property
    def ticktextsrc(self): ...
    @ticktextsrc.setter
    def ticktextsrc(self, val) -> None: ...
    @property
    def tickvals(self): ...
    @tickvals.setter
    def tickvals(self, val) -> None: ...
    @property
    def tickvalssrc(self): ...
    @tickvalssrc.setter
    def tickvalssrc(self, val) -> None: ...
    @property
    def tickwidth(self): ...
    @tickwidth.setter
    def tickwidth(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        dtick: Any | None = ...,
        exponentformat: Any | None = ...,
        labelalias: Any | None = ...,
        minexponent: Any | None = ...,
        nticks: Any | None = ...,
        range: Any | None = ...,
        separatethousands: Any | None = ...,
        showexponent: Any | None = ...,
        showticklabels: Any | None = ...,
        showtickprefix: Any | None = ...,
        showticksuffix: Any | None = ...,
        tick0: Any | None = ...,
        tickangle: Any | None = ...,
        tickcolor: Any | None = ...,
        tickfont: Any | None = ...,
        tickformat: Any | None = ...,
        tickformatstops: Any | None = ...,
        tickformatstopdefaults: Any | None = ...,
        ticklabelstep: Any | None = ...,
        ticklen: Any | None = ...,
        tickmode: Any | None = ...,
        tickprefix: Any | None = ...,
        ticks: Any | None = ...,
        ticksuffix: Any | None = ...,
        ticktext: Any | None = ...,
        ticktextsrc: Any | None = ...,
        tickvals: Any | None = ...,
        tickvalssrc: Any | None = ...,
        tickwidth: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...