from typing import Any, NamedTuple

class BinnedStatisticResult(NamedTuple):
    statistic: Any
    bin_edges: Any
    binnumber: Any

def binned_statistic(
    x, values, statistic: str = ..., bins: int = ..., range: Any | None = ...
): ...

class BinnedStatistic2dResult(NamedTuple):
    statistic: Any
    x_edge: Any
    y_edge: Any
    binnumber: Any

def binned_statistic_2d(
    x,
    y,
    values,
    statistic: str = ...,
    bins: int = ...,
    range: Any | None = ...,
    expand_binnumbers: bool = ...,
): ...

class BinnedStatisticddResult(NamedTuple):
    statistic: Any
    bin_edges: Any
    binnumber: Any

def binned_statistic_dd(
    sample,
    values,
    statistic: str = ...,
    bins: int = ...,
    range: Any | None = ...,
    expand_binnumbers: bool = ...,
    binned_statistic_result: Any | None = ...,
): ...
