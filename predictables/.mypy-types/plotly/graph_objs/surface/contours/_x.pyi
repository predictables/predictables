from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class X(_BaseTraceHierarchyType):
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def end(self): ...
    @end.setter
    def end(self, val) -> None: ...
    @property
    def highlight(self): ...
    @highlight.setter
    def highlight(self, val) -> None: ...
    @property
    def highlightcolor(self): ...
    @highlightcolor.setter
    def highlightcolor(self, val) -> None: ...
    @property
    def highlightwidth(self): ...
    @highlightwidth.setter
    def highlightwidth(self, val) -> None: ...
    @property
    def project(self): ...
    @project.setter
    def project(self, val) -> None: ...
    @property
    def show(self): ...
    @show.setter
    def show(self, val) -> None: ...
    @property
    def size(self): ...
    @size.setter
    def size(self, val) -> None: ...
    @property
    def start(self): ...
    @start.setter
    def start(self, val) -> None: ...
    @property
    def usecolormap(self): ...
    @usecolormap.setter
    def usecolormap(self, val) -> None: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        color: Any | None = ...,
        end: Any | None = ...,
        highlight: Any | None = ...,
        highlightcolor: Any | None = ...,
        highlightwidth: Any | None = ...,
        project: Any | None = ...,
        show: Any | None = ...,
        size: Any | None = ...,
        start: Any | None = ...,
        usecolormap: Any | None = ...,
        width: Any | None = ...,
        **kwargs
    ) -> None: ...
