from .base import TransformerMixin
from .utils.metaestimators import _BaseComposition
from typing import Any

class Pipeline(_BaseComposition):
    steps: Any
    memory: Any
    verbose: Any
    def __init__(
        self, steps, *, memory: Any | None = ..., verbose: bool = ...
    ) -> None: ...
    def set_output(self, *, transform: Any | None = ...): ...
    def get_params(self, deep: bool = ...): ...
    def set_params(self, **kwargs): ...
    def __len__(self): ...
    def __getitem__(self, ind): ...
    @property
    def named_steps(self): ...
    def fit(self, X, y: Any | None = ..., **fit_params): ...
    def fit_transform(self, X, y: Any | None = ..., **fit_params): ...
    def predict(self, X, **predict_params): ...
    def fit_predict(self, X, y: Any | None = ..., **fit_params): ...
    def predict_proba(self, X, **predict_proba_params): ...
    def decision_function(self, X): ...
    def score_samples(self, X): ...
    def predict_log_proba(self, X, **predict_log_proba_params): ...
    def transform(self, X): ...
    def inverse_transform(self, Xt): ...
    def score(self, X, y: Any | None = ..., sample_weight: Any | None = ...): ...
    @property
    def classes_(self): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
    @property
    def n_features_in_(self): ...
    @property
    def feature_names_in_(self): ...
    def __sklearn_is_fitted__(self): ...

def make_pipeline(*steps, memory: Any | None = ..., verbose: bool = ...): ...

class FeatureUnion(TransformerMixin, _BaseComposition):
    transformer_list: Any
    n_jobs: Any
    transformer_weights: Any
    verbose: Any
    def __init__(
        self,
        transformer_list,
        *,
        n_jobs: Any | None = ...,
        transformer_weights: Any | None = ...,
        verbose: bool = ...
    ) -> None: ...
    def set_output(self, *, transform: Any | None = ...): ...
    @property
    def named_transformers(self): ...
    def get_params(self, deep: bool = ...): ...
    def set_params(self, **kwargs): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
    def fit(self, X, y: Any | None = ..., **fit_params): ...
    def fit_transform(self, X, y: Any | None = ..., **fit_params): ...
    def transform(self, X): ...
    @property
    def n_features_in_(self): ...
    @property
    def feature_names_in_(self): ...
    def __sklearn_is_fitted__(self): ...
    def __getitem__(self, name): ...

def make_union(*transformers, n_jobs: Any | None = ..., verbose: bool = ...): ...
