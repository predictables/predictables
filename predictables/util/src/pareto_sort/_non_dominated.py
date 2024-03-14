from typing import List

import numpy as np

from predictables.util.pareto_sort._is_dominated import is_dominated  # type: ignore


def non_dominated(variables: List[np.ndarray]) -> List[np.ndarray]:
    """
    Find the non-dominated front among a list of variables.

    Parameters
    ----------
    variables : list of np.array
        A list where each element is an array representing a variable with its
        objectives.

    Returns
    -------
    list of np.array
        A list of variables that are non-dominated.
    """
    return [a for a in variables if not any(is_dominated(a, b) for b in variables)]