from .deprecation import deprecated as deprecated
from numpy import (
    ComplexWarning as ComplexWarning,
    VisibleDeprecationWarning as VisibleDeprecationWarning,
)
from scipy.integrate import trapz as trapezoid
from scipy.optimize.linesearch import (
    line_search_wolfe1 as line_search_wolfe1,
    line_search_wolfe2 as line_search_wolfe2,
)
from typing import Any

np_version: Any
np_base_version: Any
sp_version: Any
sp_base_version: Any
percentile: Any

def threadpool_limits(limits: Any | None = ..., user_api: Any | None = ...): ...
def threadpool_info(): ...
def delayed(function): ...
