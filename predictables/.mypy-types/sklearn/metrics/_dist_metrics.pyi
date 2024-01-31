from typing import Any, ClassVar

import scipy.sparse._csr
import scipy.sparse._matrix
import sklearn.externals._packaging.version

BOOL_METRICS: list
METRIC_MAPPING32: dict
METRIC_MAPPING64: dict
check_array: function
issparse: function
parse_version: function
sp_base_version: sklearn.externals._packaging.version.Version

class BrayCurtisDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class BrayCurtisDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class CanberraDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class CanberraDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class ChebyshevDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...

class ChebyshevDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...

class DiceDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class DiceDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class DistanceMetric:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    @classmethod
    def get_metric(cls, *args, **kwargs) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class DistanceMetric32(DistanceMetric):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _pairwise_dense_dense(self, *args, **kwargs) -> Any: ...
    def _pairwise_dense_sparse(self, *args, **kwargs) -> Any: ...
    def _pairwise_sparse_dense(self, *args, **kwargs) -> Any: ...
    def _pairwise_sparse_sparse(self, *args, **kwargs) -> Any: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    @classmethod
    def get_metric(cls, *args, **kwargs) -> Any: ...
    def pairwise(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...
    def __getstate__(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class DistanceMetric64(DistanceMetric):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _pairwise_dense_dense(self, *args, **kwargs) -> Any: ...
    def _pairwise_dense_sparse(self, *args, **kwargs) -> Any: ...
    def _pairwise_sparse_dense(self, *args, **kwargs) -> Any: ...
    def _pairwise_sparse_sparse(self, *args, **kwargs) -> Any: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    @classmethod
    def get_metric(cls, *args, **kwargs) -> Any: ...
    def pairwise(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...
    def __getstate__(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class EuclideanDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class EuclideanDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class HammingDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class HammingDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class HaversineDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class HaversineDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class JaccardDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class JaccardDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class KulsinskiDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class KulsinskiDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class MahalanobisDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class MahalanobisDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ManhattanDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...

class ManhattanDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...

class MatchingDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class MatchingDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class MinkowskiDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class MinkowskiDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class PyFuncDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...

class PyFuncDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...

class RogersTanimotoDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class RogersTanimotoDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class RussellRaoDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class RussellRaoDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class SEuclideanDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class SEuclideanDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _validate_data(self, *args, **kwargs) -> Any: ...
    def dist_to_rdist(self, *args, **kwargs) -> Any: ...
    def rdist_to_dist(self, *args, **kwargs) -> Any: ...

class SokalMichenerDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class SokalMichenerDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class SokalSneathDistance32(DistanceMetric32):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class SokalSneathDistance64(DistanceMetric64):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

class csr_matrix(scipy.sparse._matrix.spmatrix, scipy.sparse._csr._csr_base): ...

def __pyx_unpickle_DistanceMetric(*args, **kwargs) -> Any: ...
def __pyx_unpickle_Enum(*args, **kwargs) -> Any: ...
def get_valid_metric_ids(*args, **kwargs) -> Any: ...
def newObj(*args, **kwargs) -> Any: ...