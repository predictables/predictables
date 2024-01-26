from ..base import BaseEstimator
from ._available_if import available_if as available_if
from abc import ABCMeta, abstractmethod
from typing import Any, List

class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    steps: List[Any]
    @abstractmethod
    def __init__(self): ...
