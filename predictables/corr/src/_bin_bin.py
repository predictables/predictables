from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from predictables.util import get_column_dtype, to_pd_df, to_pd_s


def calc_binary_binary_corr(
    *args: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> Union[float, pd.DataFrame]:
    r"""
    Calculates the correlation either between two binary variables or between
    all pairs of binary variables in a data frame.

    Parameters
    ----------
    *args : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The arguments to pass to the appropriate function.

        - If one argument is passed, it is assumed to be a data
          frame or some other similar data structure.
        - If two arguments are passed, they are assumed to be two
          binary variables.

    Returns
    -------
    Union[float, pd.DataFrame]
        The correlation between two binary variables or between all pairs of
        binary variables in a data frame.

    Notes
    -----
    The phi coefficient is a measure of association between two binary variables.
    It is defined as the square root of the chi-squared statistic divided by the
    sample size:

    .. math::

        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    References
    ----------
    - https://en.wikipedia.org/wiki/Phi_coefficient
    - https://www.statisticshowto.com/phi-coefficient/
    """
    if len(args) == 1:
        return calc_binary_binary_corr_df(args[0])
    elif len(args) == 2:
        return calc_binary_binary_corr_series(args[0], args[1])
    else:
        raise TypeError(
            f"Invalid number of arguments: Must be 1 or 2, but got {len(args)}"
        )


def calc_binary_binary_corr_df(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> pd.DataFrame:
    r"""
    Calculate the correlation (phi coefficient) between binary variables
    in a data frame.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataframe containing the variables.

    Returns
    -------
    pd.DataFrame
        A correlation matrix of the binary variables.

    Notes
    -----
    The phi coefficient is a measure of association between two binary variables.
    It is defined as the square root of the chi-squared statistic divided by the
    sample size:

    .. math::

        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`\chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    It is also known as the mean square contingency coefficient.
    """
    # Ensure that the input is a pandas dataframe
    df = to_pd_df(df)

    # Get dtypes for all columns
    col_dtypes = [get_column_dtype(df[col]) for col in df.columns]

    # Selecting binary variables
    binary_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "binary"]
    ].columns.tolist()
    binary_vars = df[binary_cols]

    # Initialize an empty DataFrame to store correlation values
    corr_matrix = pd.DataFrame(index=binary_vars.columns, columns=binary_vars.columns)

    # Calculate phi coefficient for each pair of binary variables
    for col1 in binary_vars.columns:
        for col2 in binary_vars.columns:
            if col1 != col2:
                contingency_table = pd.crosstab(binary_vars[col1], binary_vars[col2])
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                n = np.sum(contingency_table.values)
                phi_coeff = np.sqrt(chi2 / (n + (chi2 == 0)))
                corr_matrix.loc[col1, col2] = phi_coeff

    # Filling diagonal with 1s (as a variable is perfectly correlated with itself)
    np.fill_diagonal(corr_matrix.values, 1)

    return corr_matrix


def calc_binary_binary_corr_series(
    s1: Union[pd.Series, pl.Series], s2: Union[pd.Series, pl.Series]
) -> float:
    r"""
    Calculate the correlation (phi coefficient) between two binary variables.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The first binary variable.
    s2 : Union[pd.Series, pl.Series]
        The second binary variable.

    Returns
    -------
    float
        The correlation between the two binary variables.

    Notes
    -----
    The phi coefficient is a measure of association between two binary variables.
    It is defined as the square root of the chi-squared statistic divided by the
    sample size:

    .. math::
        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`\chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    It is also known as the mean square contingency coefficient.

    References
    ----------
    - https://en.wikipedia.org/wiki/Phi_coefficient
    - https://www.statisticshowto.com/phi-coefficient/
    """
    # Convert to pandas series
    s1 = to_pd_s(s1)
    s2 = to_pd_s(s2)

    # Calculate phi coefficient
    contingency_table = pd.crosstab(s1, s2)
    chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table.values)
    phi_coeff = np.sqrt(chi2 / (n + (chi2 == 0)))

    return phi_coeff
