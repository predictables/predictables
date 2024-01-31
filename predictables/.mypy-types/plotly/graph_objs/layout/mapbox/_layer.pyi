from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Layer(_BaseLayoutHierarchyType):
    @property
    def below(self): ...
    @below.setter
    def below(self, val) -> None: ...
    @property
    def circle(self): ...
    @circle.setter
    def circle(self, val) -> None: ...
    @property
    def color(self): ...
    @color.setter
    def color(self, val) -> None: ...
    @property
    def coordinates(self): ...
    @coordinates.setter
    def coordinates(self, val) -> None: ...
    @property
    def fill(self): ...
    @fill.setter
    def fill(self, val) -> None: ...
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    @property
    def maxzoom(self): ...
    @maxzoom.setter
    def maxzoom(self, val) -> None: ...
    @property
    def minzoom(self): ...
    @minzoom.setter
    def minzoom(self, val) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    @property
    def source(self): ...
    @source.setter
    def source(self, val) -> None: ...
    @property
    def sourceattribution(self): ...
    @sourceattribution.setter
    def sourceattribution(self, val) -> None: ...
    @property
    def sourcelayer(self): ...
    @sourcelayer.setter
    def sourcelayer(self, val) -> None: ...
    @property
    def sourcetype(self): ...
    @sourcetype.setter
    def sourcetype(self, val) -> None: ...
    @property
    def symbol(self): ...
    @symbol.setter
    def symbol(self, val) -> None: ...
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
    def __init__(
        self,
        arg: Any | None = ...,
        below: Any | None = ...,
        circle: Any | None = ...,
        color: Any | None = ...,
        coordinates: Any | None = ...,
        fill: Any | None = ...,
        line: Any | None = ...,
        maxzoom: Any | None = ...,
        minzoom: Any | None = ...,
        name: Any | None = ...,
        opacity: Any | None = ...,
        source: Any | None = ...,
        sourceattribution: Any | None = ...,
        sourcelayer: Any | None = ...,
        sourcetype: Any | None = ...,
        symbol: Any | None = ...,
        templateitemname: Any | None = ...,
        type: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...