from ..base import (
    BaseEstimator as BaseEstimator,
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    TransformerMixin as TransformerMixin,
)
from ..utils import (
    check_random_state as check_random_state,
    gen_batches as gen_batches,
    gen_even_slices as gen_even_slices,
)
from ..utils._param_validation import Interval as Interval, StrOptions as StrOptions
from ..utils.parallel import Parallel as Parallel, delayed as delayed
from ..utils.validation import (
    check_is_fitted as check_is_fitted,
    check_non_negative as check_non_negative,
)
from typing import Any

EPS: Any

class LatentDirichletAllocation(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    n_components: Any
    doc_topic_prior: Any
    topic_word_prior: Any
    learning_method: Any
    learning_decay: Any
    learning_offset: Any
    max_iter: Any
    batch_size: Any
    evaluate_every: Any
    total_samples: Any
    perp_tol: Any
    mean_change_tol: Any
    max_doc_update_iter: Any
    n_jobs: Any
    verbose: Any
    random_state: Any
    def __init__(
        self,
        n_components: int = ...,
        *,
        doc_topic_prior: Any | None = ...,
        topic_word_prior: Any | None = ...,
        learning_method: str = ...,
        learning_decay: float = ...,
        learning_offset: float = ...,
        max_iter: int = ...,
        batch_size: int = ...,
        evaluate_every: int = ...,
        total_samples: float = ...,
        perp_tol: float = ...,
        mean_change_tol: float = ...,
        max_doc_update_iter: int = ...,
        n_jobs: Any | None = ...,
        verbose: int = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def partial_fit(self, X, y: Any | None = ...): ...
    bound_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def score(self, X, y: Any | None = ...): ...
    def perplexity(self, X, sub_sampling: bool = ...): ...
