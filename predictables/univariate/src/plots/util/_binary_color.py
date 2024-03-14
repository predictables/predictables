from typing import Union

import numpy as np


def binary_color(x: Union[int, float, str, bool]) -> str:
    """
    Return a color for each value in x, based on whether it is 0 or 1.

    Parameters
    ----------
    x : Union[int, float, str, bool]
        The value to get the color for.

    Returns
    -------
    int
        The color for x.
    """
    if isinstance(x, (np.int64, np.int32)):
        x = int(x)
    elif isinstance(x, (np.float64, np.float32)):
        x = float(x)

    if isinstance(x, (str, bool)):
        x_ = x.lower() if isinstance(x, str) else str(x).lower()
        x_ = x_.strip().replace("+", "")
        if x_ in ["0", "-1", "false", "f", "no", "n"]:
            return "blue"
        elif x_ in ["1", "true", "t", "yes", "y"]:
            return "orange"
        else:
            raise ValueError(f"Invalid value {x} for binary variable.")
    elif isinstance(x, (int, float)):
        x_1 = float(x)
        if x_1 in [0.0, -1.0]:
            return "blue"
        elif x_1 in [1.0]:
            return "orange"
        else:
            raise ValueError(f"Invalid value {x_1} for binary variable.")

    raise ValueError(f"Invalid value {x} for binary variable.")
