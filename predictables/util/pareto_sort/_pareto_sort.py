from typing import List

import numpy as np

from predictables.util.pareto_sort._is_dominated import is_dominated
from predictables.util.pareto_sort._non_dominated import non_dominated


def pareto_dominated_sort(variables: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sort a list of variables based on Pareto optimality.

    Parameters
    ----------
    variables : list of np.array
        A list where each element is an array representing a variable with its objectives.

    Returns
    -------
    list of np.array
        A sorted list of variables based on Pareto optimality.
    """
    return [
        a
        for a in variables
        if not any(is_dominated(a, b) for b in variables if not np.array_equal(a, b))
    ]


def pareto_non_dominated_sort(variables: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sort a list of variables based on Pareto optimality using non-dominated sorting.

    Parameters
    ----------
    variables : list of np.array
        A list where each element is an array representing a variable with its objectives.

    Returns
    -------
    list of list of np.array
        A sorted list of variables into different Pareto fronts.
    """
    sorted_variables = []
    while variables:
        front = non_dominated(variables)
        sorted_variables.append(front)
        variables = [v for v in variables if v not in front]
    return sorted_variables
