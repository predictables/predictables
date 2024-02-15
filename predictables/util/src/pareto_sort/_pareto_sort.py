from typing import List

import numpy as np

from predictables.util.src.pareto_sort._is_dominated import is_dominated

# from predictables.util.src.pareto_sort._non_dominated import non_dominated


def pareto_sort(variables: List[np.ndarray]) -> List[np.ndarray]:
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
    # n_objectives = variables[0].shape[0]

    # Loop over the variables, sorting based on the is_dominated condition
    sorted_variables: List[np.ndarray] = []
    sorded_order_index: List[int] = []
    for i, variable in enumerate(variables):
        if not any(is_dominated(variable, v) for v in sorted_variables):
            sorted_variables.append(variable)
            sorded_order_index.append(i)
        else:
            continue

    # Return the sorted variables
    return [variables[i] for i in sorded_order_index]


# def pareto_non_dominated_sort(variables: List[np.ndarray]) -> List[np.ndarray]:
#     """
#     Sort a list of variables based on Pareto optimality using non-dominated sorting.

#     Parameters
#     ----------
#     variables : list of np.array
#         A list where each element is an array representing a variable with
# its objectives.

#     Returns
#     -------
#     list of list of np.array
#         A sorted list of variables into different Pareto fronts.
#     """
#     sorted_variables = []
#     while variables:
#         front = non_dominated(variables)
#         sorted_variables.append(front)
#         variables = [v for v in variables if v not in front]
#     return sorted_variables
