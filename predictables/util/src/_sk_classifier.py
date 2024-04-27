from typing import Protocol

class SKClassifier(Protocol):
    """Indicate that a type should be any class implementing a fit and a predict method."""
    def fit(self, X, y, **kwargs):
        ...

    def predict(self, X):
        ...