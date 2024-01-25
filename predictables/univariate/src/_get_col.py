from typing import Dict, List, Union

import pandas as pd


def get_col(
    col: str, cv: Dict[int, float], unique_folds: List[int]
) -> List[Union[int, float, str]]:
    """
    Helper function to get the requested column from the data.

    Parameters
    ----------
    col : str
        The name of the column to get. Choices are
            - "coef"
            - "pvalues",
            - "se"
            - "lower_ci"
            - "upper_ci"
            - "acc_train"
            - "acc_test"
            - "auc_train"
            - "auc_test"
            - "f1_train"
            - "f1_test"
            - "precision_train"
            - "precision_test"
            - "recall_train"
            - "recall_test"
            - "mcc_train"
            - "mcc_test"
            - "logloss_train"
            - "logloss_test"
            - "auc_train"
            - "auc_test"

    cv : Dict[int, float]
        The cross validation results.


    Returns
    -------
    List[Union[int, float, str]]
        The values for the requested column.

    Examples
    --------
    # Assume you have fit a model with coefficients [0.1, 0.2, 0.3, 0.4, 0.5]
    # and a standard deviation of 0.01.

    >>> get_col(self, "coef")
    [0.1, 0.2, 0.3, 0.4, 0.5]

    >>> get_col(self, "std")
    [0.01, 0.01, 0.01, 0.01, 0.01]
    """

    attributes = [getattr(self.cv[fold], col) for fold in self.unique_folds]
    sd = pd.Series(attributes).std()

    return attributes + [getattr(self, col)] + [sd]
