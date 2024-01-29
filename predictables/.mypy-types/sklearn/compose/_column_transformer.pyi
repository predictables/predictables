from ..base import TransformerMixin
from ..utils.metaestimators import _BaseComposition
from typing import Any

class ColumnTransformer(TransformerMixin, _BaseComposition):
    transformers: Any
    remainder: Any
    sparse_threshold: Any
    n_jobs: Any
    transformer_weights: Any
    verbose: Any
    verbose_feature_names_out: Any
    def __init__(
        self,
        transformers,
        *,
        remainder: str = ...,
        sparse_threshold: float = ...,
        n_jobs: Any | None = ...,
        transformer_weights: Any | None = ...,
        verbose: bool = ...,
        verbose_feature_names_out: bool = ...
    ) -> None: ...
    def set_output(self, *, transform: Any | None = ...): ...
    def get_params(self, deep: bool = ...): ...
    def set_params(self, **kwargs): ...
    @property
    def named_transformers_(self): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
    def fit(self, X, y: Any | None = ...): ...
    sparse_output_: Any
    def fit_transform(self, X, y: Any | None = ...): ...
    def transform(self, X): ...

def make_column_transformer(
    *transformers,
    remainder: str = ...,
    sparse_threshold: float = ...,
    n_jobs: Any | None = ...,
    verbose: bool = ...,
    verbose_feature_names_out: bool = ...
): ...

class make_column_selector:
    pattern: Any
    dtype_include: Any
    dtype_exclude: Any
    def __init__(
        self,
        pattern: Any | None = ...,
        *,
        dtype_include: Any | None = ...,
        dtype_exclude: Any | None = ...
    ) -> None: ...
    def __call__(self, df): ...
