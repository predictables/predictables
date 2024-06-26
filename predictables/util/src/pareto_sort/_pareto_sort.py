from __future__ import annotations

import numpy as np

from predictables.util.src.pareto_sort._is_dominated import is_dominated


def pareto_sort(variables: list[np.ndarray]) -> list[np.ndarray]:
    """
    Sort a list of variables based on Pareto optimality.

    Parameters
    ----------
    variables : list of np.array
        A list where each element is an array representing a variable
        with its objectives.

    Returns
    -------
    list of np.array
        A sorted list of variables based on Pareto optimality.
    """
    # Loop over the variables, sorting based on the is_dominated condition
    sorted_variables: list[np.ndarray] = []
    sorded_order_index: list[int] = []
    for i, variable in enumerate(variables):
        if not any(is_dominated(variable, v) for v in sorted_variables):
            sorted_variables.append(variable)
            sorded_order_index.append(i)
        else:
            continue

    # Return the sorted variables
    return [variables[i] for i in sorded_order_index]
