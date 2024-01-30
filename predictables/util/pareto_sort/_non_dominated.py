from typing import List

import numpy as np

from predictables.util.pareto_sort._is_dominated import is_dominated


def non_dominated(variables: List[np.ndarray]) -> List[np.ndarray]:
    """
    Find the non-dominated front among a list of variables.

    Parameters
    ----------
    variables : list of np.array
        A list where each element is an array representing a variable with its objectives.

    Returns
    -------
    list of np.array
        A list of variables that are non-dominated.
    """
    non_dominated = []
    for i, a in enumerate(variables):
        if not any(is_dominated(a, b) for j, b in enumerate(variables) if i != j):
            non_dominated.append(a)
    return non_dominated
