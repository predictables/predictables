import numpy as np
from typing import Tuple


def reconcile_train_test_val_sizes(
    train_size: float = None, test_size: float = None, val_size: float = None
) -> Tuple[float, float, float]:
    """
    Reconciles the sizes of the training, testing, and validation datasets,
    by either truncating or padding them.

    If any of the train_size, test_size, or val_size attributes are None, they
    will be automatically computed based on the other two attributes, using the
    following rules:
        - If only two attributes are provided, the third one will be set to the
            remaining fraction of the dataset.
        - If only one attribute is provided, the other two will be set to equal
            fractions of the dataset.

    If all three attributes are provided, they will be checked for consistency,
    and an error will be raised if they do not sum up to 1.

    Parameters
    ----------
    train_size : float, optional
        The fraction of the dataset to be used for training, by default None
    test_size : float, optional
        The fraction of the dataset to be used for testing, by default None
    val_size : float, optional
        The fraction of the dataset to be used for validation, by default None

    Returns
    -------
    Tuple[float, float, float]
        The reconciled train, test, and val sizes.

    Raises
    ------
    ValueError
        If the provided train, test, and val sizes do not sum up to 1.
    """
    sizes = [train_size, test_size, val_size]
    none_count = sizes.count(None)

    if none_count == 3:
        return 0.7, 0.2, 0.1
    elif none_count == 2:
        defined_size = next(size for size in sizes if size is not None)
        other_sizes = [(1 - defined_size) / 2] * 2
        return tuple(size if size is not None else other_sizes.pop() for size in sizes)
    elif none_count == 1:
        non_none_sizes = [size for size in sizes if size is not None]
        none_index = sizes.index(None)
        sizes[none_index] = 1 - sum(non_none_sizes)
        return tuple(sizes)
    else:
        if np.isclose(sum(sizes), 1.0):
            return tuple(sizes)
        else:
            raise ValueError(
                f"Train, val, and test sizes must sum to 1.0. Got: {sizes}"
            )
