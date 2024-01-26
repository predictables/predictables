from typing import Any, ClassVar

import numpy.dtypes

CLASS_DOC: str
DOC_DICT: dict
NodeData: numpy.dtypes.VoidDType
NodeHeapData: numpy.dtypes.VoidDType
VALID_METRICS: list
VALID_METRIC_IDS: list
check_array: function

class BinaryTree:
    data: ClassVar[getset_descriptor] = ...
    idx_array: ClassVar[getset_descriptor] = ...
    node_bounds: ClassVar[getset_descriptor] = ...
    node_data: ClassVar[getset_descriptor] = ...
    sample_weight: ClassVar[getset_descriptor] = ...
    sum_weight: ClassVar[getset_descriptor] = ...
    valid_metrics: ClassVar[list] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _update_sample_weight(self, *args, **kwargs) -> Any: ...
    def get_arrays(self) -> Any: ...
    def get_n_calls(self) -> Any: ...
    def get_tree_stats(self) -> Any: ...
    def kernel_density(
        self, X, h, kernel=..., atol=..., rtol=..., breadth_first=..., return_log=...
    ) -> Any: ...
    def query(
        self, X, k=..., return_distance=..., dualtree=..., breadth_first=...
    ) -> Any: ...
    def query_radius(
        self, X, r, return_distance=..., count_only=..., sort_results=...
    ) -> Any: ...
    def reset_n_calls(self) -> Any: ...
    def two_point_correlation(self, X, r, dualtree=...) -> Any: ...
    def __getstate__(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class KDTree(BinaryTree):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class NeighborsHeap:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def get_arrays(self, *args, **kwargs) -> Any: ...
    def push(self, *args, **kwargs) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class NodeHeap:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

def __pyx_unpickle_Enum(*args, **kwargs) -> Any: ...
def get_valid_metric_ids(*args, **kwargs) -> Any: ...
def kernel_norm(*args, **kwargs) -> Any: ...
def load_heap(*args, **kwargs) -> Any: ...
def newObj(*args, **kwargs) -> Any: ...
def nodeheap_sort(*args, **kwargs) -> Any: ...
def simultaneous_sort(*args, **kwargs) -> Any: ...
