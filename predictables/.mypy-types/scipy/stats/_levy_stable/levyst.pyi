from typing import Any, ClassVar

class Nolan:
    c1: ClassVar[getset_descriptor] = ...
    c2: ClassVar[getset_descriptor] = ...
    c3: ClassVar[getset_descriptor] = ...
    xi: ClassVar[getset_descriptor] = ...
    zeta: ClassVar[getset_descriptor] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def g(self, *args, **kwargs) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...
