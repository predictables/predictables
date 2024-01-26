from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Stream(_BaseTraceHierarchyType):
    @property
    def maxpoints(self): ...
    @maxpoints.setter
    def maxpoints(self, val) -> None: ...
    @property
    def token(self): ...
    @token.setter
    def token(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        maxpoints: Any | None = ...,
        token: Any | None = ...,
        **kwargs
    ) -> None: ...
