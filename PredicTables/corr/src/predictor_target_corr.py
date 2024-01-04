from .ctsX_ctsY import calc_continuousX_continuousY_corr
from .ctsX_binY import calc_continuousX_binaryY_corr
from .ctsX_catY import calc_continuousX_categoricalY_corr
from .binX_binY import calc_binaryX_binaryY_corr
from .binX_catY import calc_binaryX_categoricalY_corr
from .catX_catY import calc_categoricalX_categoricalY_corr

from PredicTables.util import get_column_dtype


def predictor_target_corr(X, y):
    X_type = get_column_dtype(X)
    y_type = get_column_dtype(y)

    if X_type == "continuous" and y_type == "continuous":
        return calc_continuousX_continuousY_corr(X, y)
    elif X_type == "continuous" and y_type == "binary":
        return calc_continuousX_binaryY_corr(X, y)
    elif X_type == "continuous" and y_type == "categorical":
        return calc_continuousX_categoricalY_corr(X, y)
    elif X_type == "binary" and y_type == "binary":
        return calc_binaryX_binaryY_corr(X, y)
    elif X_type == "binary" and y_type == "categorical":
        return calc_binaryX_categoricalY_corr(X, y)
    elif X_type == "categorical" and y_type == "categorical":
        return calc_categoricalX_categoricalY_corr(X, y)
    else:
        raise TypeError(f"Unknown dtype: X={X_type}, y={y_type}")
