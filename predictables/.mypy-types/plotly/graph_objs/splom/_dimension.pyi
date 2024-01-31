from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Dimension(_BaseTraceHierarchyType):
    @property
    def axis(self): ...
    @axis.setter
    def axis(self, val) -> None: ...
    @property
    def label(self): ...
    @label.setter
    def label(self, val) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def templateitemname(self): ...
    @templateitemname.setter
    def templateitemname(self, val) -> None: ...
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
    def __init__(
        self,
        arg: Any | None = ...,
        axis: Any | None = ...,
        label: Any | None = ...,
        name: Any | None = ...,
        templateitemname: Any | None = ...,
        values: Any | None = ...,
        valuessrc: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...