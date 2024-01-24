from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import scipy

from predictables.src._correlation_matrix import _correlation_matrix


def _eigenvalues(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame, np.ndarray], round_to: int = 4
) -> pd.DataFrame:
    """
    Calculate the eigenvalues of the correlation matrix of the given data, then
    a condition index. The condition index is the ratio of the largest eigenvalue
    to each of the other eigenvalues. The larger the ratio, the larger the
    condition number, the more likely it is that the correlation matrix is
    ill-conditioned.

    In this implementation, I am displaying the log of the condition number
    because the condition number is often very large, and we are more interested
    in order of magnitude than the exact value.

    The output is sorted by the condition number, so that the most ill-conditioned,
    and thus most likely to cause problems due to multicollinearity, features are
    at the top of the DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame,
                 pl.DataFrame,
                 pl.LazyFrame,
                 np.ndarray]
        The data to calculate the eigenvalues of the correlation matrix of.
    round_to : int, optional
        The number of decimals to round the eigenvalues to, by default 4.

    Returns
    -------
    pd.DataFrame
        The features, eigenvalues, and condition numbers of the data, sorted by
        condition number.
    """
    # Calculate the correlation matrix
    corr_matrix = _correlation_matrix(data)

    # Get the feature names from the original data
    if isinstance(data, pd.DataFrame):
        feature_names = data.columns.tolist()
    elif isinstance(data, pl.DataFrame):
        feature_names = data.columns
    elif isinstance(data, pl.LazyFrame):
        feature_names = data.columns
    else:
        feature_names = [f"x{i}" for i in range(data.shape[1])]

    # Calculate the eigenvalues of the correlation matrix
    eigenvalues = scipy.linalg.eig(corr_matrix)[0]

    # Create a DataFrame with the eigenvalues
    df = pd.DataFrame(
        {"feature": feature_names, "eigenvalue": eigenvalues.real},
        columns=["feature", "eigenvalue"],
    )

    # Calculate a condition number
    df["log[condition_number]"] = np.log(df["eigenvalue"].max() / df["eigenvalue"])

    # Round the eigenvalues
    df["eigenvalue"] = df["eigenvalue"].round(round_to)
    df["log[condition_number]"] = df["log[condition_number]"].round(round_to)

    # Sort the DataFrame by the condition number
    df = df.sort_values(by="log[condition_number]", ascending=False).reset_index(
        drop=True
    )

    return df
