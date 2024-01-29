from ..base import BaseEstimator, ClassifierMixin, MultiOutputMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
from typing import Any

class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    criterion: Any
    splitter: Any
    max_depth: Any
    min_samples_split: Any
    min_samples_leaf: Any
    min_weight_fraction_leaf: Any
    max_features: Any
    max_leaf_nodes: Any
    random_state: Any
    min_impurity_decrease: Any
    class_weight: Any
    ccp_alpha: Any
    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight: Any | None = ...,
        ccp_alpha: float = ...
    ): ...
    def get_depth(self): ...
    def get_n_leaves(self): ...
    def predict(self, X, check_input: bool = ...): ...
    def apply(self, X, check_input: bool = ...): ...
    def decision_path(self, X, check_input: bool = ...): ...
    def cost_complexity_pruning_path(self, X, y, sample_weight: Any | None = ...): ...
    @property
    def feature_importances_(self): ...

class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    def __init__(
        self,
        *,
        criterion: str = ...,
        splitter: str = ...,
        max_depth: Any | None = ...,
        min_samples_split: int = ...,
        min_samples_leaf: int = ...,
        min_weight_fraction_leaf: float = ...,
        max_features: Any | None = ...,
        random_state: Any | None = ...,
        max_leaf_nodes: Any | None = ...,
        min_impurity_decrease: float = ...,
        class_weight: Any | None = ...,
        ccp_alpha: float = ...
    ) -> None: ...
    def fit(self, X, y, sample_weight: Any | None = ..., check_input: bool = ...): ...
    def predict_proba(self, X, check_input: bool = ...): ...
    def predict_log_proba(self, X): ...

class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    def __init__(
        self,
        *,
        criterion: str = ...,
        splitter: str = ...,
        max_depth: Any | None = ...,
        min_samples_split: int = ...,
        min_samples_leaf: int = ...,
        min_weight_fraction_leaf: float = ...,
        max_features: Any | None = ...,
        random_state: Any | None = ...,
        max_leaf_nodes: Any | None = ...,
        min_impurity_decrease: float = ...,
        ccp_alpha: float = ...
    ) -> None: ...
    def fit(self, X, y, sample_weight: Any | None = ..., check_input: bool = ...): ...

class ExtraTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        *,
        criterion: str = ...,
        splitter: str = ...,
        max_depth: Any | None = ...,
        min_samples_split: int = ...,
        min_samples_leaf: int = ...,
        min_weight_fraction_leaf: float = ...,
        max_features: str = ...,
        random_state: Any | None = ...,
        max_leaf_nodes: Any | None = ...,
        min_impurity_decrease: float = ...,
        class_weight: Any | None = ...,
        ccp_alpha: float = ...
    ) -> None: ...

class ExtraTreeRegressor(DecisionTreeRegressor):
    def __init__(
        self,
        *,
        criterion: str = ...,
        splitter: str = ...,
        max_depth: Any | None = ...,
        min_samples_split: int = ...,
        min_samples_leaf: int = ...,
        min_weight_fraction_leaf: float = ...,
        max_features: float = ...,
        random_state: Any | None = ...,
        min_impurity_decrease: float = ...,
        max_leaf_nodes: Any | None = ...,
        ccp_alpha: float = ...
    ) -> None: ...
