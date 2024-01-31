from ..utils.metadata_routing import _MetadataRequester
from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any

class GroupsConsumerMixin(_MetadataRequester): ...

class BaseCrossValidator(_MetadataRequester, metaclass=ABCMeta):
    def split(
        self, X, y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...
    @abstractmethod
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...

class LeaveOneOut(BaseCrossValidator):
    def get_n_splits(self, X, y: Any | None = ..., groups: Any | None = ...): ...

class LeavePOut(BaseCrossValidator):
    p: Any
    def __init__(self, p) -> None: ...
    def get_n_splits(self, X, y: Any | None = ..., groups: Any | None = ...): ...

class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    n_splits: Any
    shuffle: Any
    random_state: Any
    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state): ...
    def split(
        self, X, y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...

class KFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        shuffle: bool = ...,
        random_state: Any | None = ...
    ) -> None: ...

class GroupKFold(GroupsConsumerMixin, _BaseKFold):
    def __init__(self, n_splits: int = ...) -> None: ...
    def split(self, X, y: Any | None = ..., groups: Any | None = ...): ...

class StratifiedKFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        shuffle: bool = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def split(self, X, y, groups: Any | None = ...): ...

class StratifiedGroupKFold(GroupsConsumerMixin, _BaseKFold):
    def __init__(
        self, n_splits: int = ..., shuffle: bool = ..., random_state: Any | None = ...
    ) -> None: ...

class TimeSeriesSplit(_BaseKFold):
    max_train_size: Any
    test_size: Any
    gap: Any
    def __init__(
        self,
        n_splits: int = ...,
        *,
        max_train_size: Any | None = ...,
        test_size: Any | None = ...,
        gap: int = ...
    ) -> None: ...
    def split(
        self, X, y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...

class LeaveOneGroupOut(GroupsConsumerMixin, BaseCrossValidator):
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...
    def split(self, X, y: Any | None = ..., groups: Any | None = ...): ...

class LeavePGroupsOut(GroupsConsumerMixin, BaseCrossValidator):
    n_groups: Any
    def __init__(self, n_groups) -> None: ...
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...
    def split(self, X, y: Any | None = ..., groups: Any | None = ...): ...

class _RepeatedSplits(_MetadataRequester, metaclass=ABCMeta):
    cv: Any
    n_repeats: Any
    random_state: Any
    cvargs: Any
    def __init__(
        self, cv, *, n_repeats: int = ..., random_state: Any | None = ..., **cvargs
    ) -> None: ...
    def split(
        self, X, y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...

class RepeatedKFold(_RepeatedSplits):
    def __init__(
        self,
        *,
        n_splits: int = ...,
        n_repeats: int = ...,
        random_state: Any | None = ...
    ) -> None: ...

class RepeatedStratifiedKFold(_RepeatedSplits):
    def __init__(
        self,
        *,
        n_splits: int = ...,
        n_repeats: int = ...,
        random_state: Any | None = ...
    ) -> None: ...

class BaseShuffleSplit(_MetadataRequester, metaclass=ABCMeta):
    n_splits: Any
    test_size: Any
    train_size: Any
    random_state: Any
    def __init__(
        self,
        n_splits: int = ...,
        *,
        test_size: Any | None = ...,
        train_size: Any | None = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def split(
        self, X, y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...

class ShuffleSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        test_size: Any | None = ...,
        train_size: Any | None = ...,
        random_state: Any | None = ...
    ) -> None: ...

class GroupShuffleSplit(GroupsConsumerMixin, ShuffleSplit):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        test_size: Any | None = ...,
        train_size: Any | None = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def split(self, X, y: Any | None = ..., groups: Any | None = ...): ...

class StratifiedShuffleSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        test_size: Any | None = ...,
        train_size: Any | None = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def split(self, X, y, groups: Any | None = ...): ...

class PredefinedSplit(BaseCrossValidator):
    test_fold: Any
    unique_folds: Any
    def __init__(self, test_fold) -> None: ...
    def split(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...

class _CVIterableWrapper(BaseCrossValidator):
    cv: Any
    def __init__(self, cv) -> None: ...
    def get_n_splits(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ): ...
    def split(
        self, X: Any | None = ..., y: Any | None = ..., groups: Any | None = ...
    ) -> Generator[Any, None, None]: ...

def check_cv(cv: int = ..., y: Any | None = ..., *, classifier: bool = ...): ...
def train_test_split(
    *arrays,
    test_size: Any | None = ...,
    train_size: Any | None = ...,
    random_state: Any | None = ...,
    shuffle: bool = ...,
    stratify: Any | None = ...
): ...