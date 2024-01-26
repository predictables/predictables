from ..base import TransformerMixin as TransformerMixin
from ..utils import metadata_routing as metadata_routing
from ..utils.validation import check_is_fitted as check_is_fitted
from typing import Any

class AgglomerationTransform(TransformerMixin):
    def transform(self, X): ...
    def inverse_transform(self, Xt: Any | None = ..., Xred: Any | None = ...): ...
