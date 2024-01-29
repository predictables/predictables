import numpy as np

def entropy(
    pk: np.typing.ArrayLike,
    qk: Union[np.typing.ArrayLike, None] = ...,
    base: Union[float, None] = ...,
    axis: int = ...,
) -> Union[np.number, np.ndarray]: ...
def differential_entropy(
    values: np.typing.ArrayLike,
    *,
    window_length: Union[int, None] = ...,
    base: Union[float, None] = ...,
    axis: int = ...,
    method: str = ...
) -> Union[np.number, np.ndarray]: ...
