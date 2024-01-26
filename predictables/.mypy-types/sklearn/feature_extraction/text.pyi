from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from ._stop_words import ENGLISH_STOP_WORDS as ENGLISH_STOP_WORDS
from typing import Any

def strip_accents_unicode(s): ...
def strip_accents_ascii(s): ...
def strip_tags(s): ...

class _VectorizerMixin:
    def decode(self, doc): ...
    def build_preprocessor(self): ...
    def build_tokenizer(self): ...
    def get_stop_words(self): ...
    def build_analyzer(self): ...

class HashingVectorizer(TransformerMixin, _VectorizerMixin, BaseEstimator):
    input: Any
    encoding: Any
    decode_error: Any
    strip_accents: Any
    preprocessor: Any
    tokenizer: Any
    analyzer: Any
    lowercase: Any
    token_pattern: Any
    stop_words: Any
    n_features: Any
    ngram_range: Any
    binary: Any
    norm: Any
    alternate_sign: Any
    dtype: Any
    def __init__(
        self,
        *,
        input: str = ...,
        encoding: str = ...,
        decode_error: str = ...,
        strip_accents: Any | None = ...,
        lowercase: bool = ...,
        preprocessor: Any | None = ...,
        tokenizer: Any | None = ...,
        stop_words: Any | None = ...,
        token_pattern: str = ...,
        ngram_range=...,
        analyzer: str = ...,
        n_features=...,
        binary: bool = ...,
        norm: str = ...,
        alternate_sign: bool = ...,
        dtype=...
    ) -> None: ...
    def partial_fit(self, X, y: Any | None = ...): ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def fit_transform(self, X, y: Any | None = ...): ...

class CountVectorizer(_VectorizerMixin, BaseEstimator):
    input: Any
    encoding: Any
    decode_error: Any
    strip_accents: Any
    preprocessor: Any
    tokenizer: Any
    analyzer: Any
    lowercase: Any
    token_pattern: Any
    stop_words: Any
    max_df: Any
    min_df: Any
    max_features: Any
    ngram_range: Any
    vocabulary: Any
    binary: Any
    dtype: Any
    def __init__(
        self,
        *,
        input: str = ...,
        encoding: str = ...,
        decode_error: str = ...,
        strip_accents: Any | None = ...,
        lowercase: bool = ...,
        preprocessor: Any | None = ...,
        tokenizer: Any | None = ...,
        stop_words: Any | None = ...,
        token_pattern: str = ...,
        ngram_range=...,
        analyzer: str = ...,
        max_df: float = ...,
        min_df: int = ...,
        max_features: Any | None = ...,
        vocabulary: Any | None = ...,
        binary: bool = ...,
        dtype=...
    ) -> None: ...
    def fit(self, raw_documents, y: Any | None = ...): ...
    vocabulary_: Any
    def fit_transform(self, raw_documents, y: Any | None = ...): ...
    def transform(self, raw_documents): ...
    def inverse_transform(self, X): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...

class TfidfTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    norm: Any
    use_idf: Any
    smooth_idf: Any
    sublinear_tf: Any
    def __init__(
        self,
        *,
        norm: str = ...,
        use_idf: bool = ...,
        smooth_idf: bool = ...,
        sublinear_tf: bool = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X, copy: bool = ...): ...
    @property
    def idf_(self): ...
    @idf_.setter
    def idf_(self, value) -> None: ...

class TfidfVectorizer(CountVectorizer):
    norm: Any
    use_idf: Any
    smooth_idf: Any
    sublinear_tf: Any
    def __init__(
        self,
        *,
        input: str = ...,
        encoding: str = ...,
        decode_error: str = ...,
        strip_accents: Any | None = ...,
        lowercase: bool = ...,
        preprocessor: Any | None = ...,
        tokenizer: Any | None = ...,
        analyzer: str = ...,
        stop_words: Any | None = ...,
        token_pattern: str = ...,
        ngram_range=...,
        max_df: float = ...,
        min_df: int = ...,
        max_features: Any | None = ...,
        vocabulary: Any | None = ...,
        binary: bool = ...,
        dtype=...,
        norm: str = ...,
        use_idf: bool = ...,
        smooth_idf: bool = ...,
        sublinear_tf: bool = ...
    ) -> None: ...
    @property
    def idf_(self): ...
    @idf_.setter
    def idf_(self, value) -> None: ...
    def fit(self, raw_documents, y: Any | None = ...): ...
    def fit_transform(self, raw_documents, y: Any | None = ...): ...
    def transform(self, raw_documents): ...
