from typing import Any, ClassVar

class StdVectorSentinel:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class StdVectorSentinelFloat64(StdVectorSentinel):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class StdVectorSentinelInt32(StdVectorSentinel):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class StdVectorSentinelInt64(StdVectorSentinel):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class StdVectorSentinelIntP(StdVectorSentinel):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

def __pyx_unpickle_StdVectorSentinel(*args, **kwargs) -> Any: ...
def __pyx_unpickle_StdVectorSentinelFloat64(*args, **kwargs) -> Any: ...
def __pyx_unpickle_StdVectorSentinelInt32(*args, **kwargs) -> Any: ...
def __pyx_unpickle_StdVectorSentinelInt64(*args, **kwargs) -> Any: ...
def __pyx_unpickle_StdVectorSentinelIntP(*args, **kwargs) -> Any: ...
