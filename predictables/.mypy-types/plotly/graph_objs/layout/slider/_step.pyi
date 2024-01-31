from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Step(_BaseLayoutHierarchyType):
    @property
    def args(self): ...
    @args.setter
    def args(self, val) -> None: ...
    @property
    def execute(self): ...
    @execute.setter
    def execute(self, val) -> None: ...
    @property
    def label(self): ...
    @label.setter
    def label(self, val) -> None: ...
    @property
    def method(self): ...
    @method.setter
    def method(self, val) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, val) -> None: ...
    @property
    def templateitemname(self): ...
    @templateitemname.setter
    def templateitemname(self, val) -> None: ...
    @property
    def value(self): ...
    @value.setter
    def value(self, val) -> None: ...
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        args: Any | None = ...,
        execute: Any | None = ...,
        label: Any | None = ...,
        method: Any | None = ...,
        name: Any | None = ...,
        templateitemname: Any | None = ...,
        value: Any | None = ...,
        visible: Any | None = ...,
        **kwargs
    ) -> None: ...