from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats  # type: ignore

from predictables.util import get_column_dtype, to_pd_df, to_pd_s


def calc_categorical_categorical_corr(
    *args: Union[pd.Series, pd.DataFrame],
) -> Union[float, pd.DataFrame]:
    r"""
    Calculates the correlation either between two categorical variables or between
    all pairs of categorical variables in a data frame.

    Parameters
    ----------
    *args : Union[pd.Series, pd.DataFrame]
        The arguments to pass to the appropriate function.

        - If one argument is passed, it is assumed to be a data
          frame or some other similar data structure.
        - If two arguments are passed, they are assumed to be two
          categorical variables.

    Returns
    -------
    Union[float, pd.DataFrame]
        The correlation between two categorical variables or between all pairs of
        categorical variables in a data frame.

    Notes
    -----
    Cramér's :math:`V` is a measure of association between two categorical variables
    and ranges from :math:`0` to :math:`+1` (inclusive).

    Cramér's :math:`V` is defined as the square root of the chi-squared statistic
    divided by the sample size:

    .. math::

        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`\chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    It is also known as the mean square contingency coefficient. It is a
    generalization of the phi coefficient, which is used when both variables
    are binary. In particular, in the case of a 2x2 contingency table,
    Cramér's :math:`V` is equivalent to the absolute value of the phi coefficient.

    Cramér's :math:`V` is symmetric. That is, the correlation between variable
    :math:`A` and variable :math:`B` is the same as the correlation between variable
    :math:`B` and variable :math:`A`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    - https://www.statisticshowto.com/v-statistics
    """
    if len(args) == 1:
        return calc_categorical_categorical_corr_df(pd.DataFrame(args[0]))
    elif len(args) == 2:
        arg1 = to_pd_s(args[0]) if isinstance(args[0], pl.Series) else args[0]  # type: ignore
        return calc_categorical_categorical_corr_series(arg1, args[1])  # type: ignore
    else:
        raise TypeError(
            f"Invalid number of arguments: Must be 1 or 2, but got {len(args)}"
        )


# Defining the function to calculate categorical-categorical correlation
def calc_categorical_categorical_corr_df(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> pd.DataFrame:
    r"""
    Calculate the correlation (Cramér's V) between categorical variables in a df.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The df containing the categorical variables.

    Returns
    -------
    pd.DataFrame
        A correlation matrix of the categorical variables.

    Notes
    -----
    Cramér's :math:`V` is a measure of association between two categorical variables
    and ranges from :math:`0` to :math:`+1` (inclusive).

    Cramér's :math:`V` is defined as the square root of the chi-squared statistic
    divided by the sample size:

    .. math::

        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`\chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    It is also known as the mean square contingency coefficient. It is a
    generalization of the phi coefficient, which is used when both variables
    are binary. In particular, in the case of a 2x2 contingency table,
    Cramér's :math:`V` is equivalent to the absolute value of the phi coefficient.

    Cramér's :math:`V` is symmetric. That is, the correlation between variable
    :math:`A` and variable :math:`B` is the same as the correlation between variable
    :math:`B` and variable :math:`A`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    - https://www.statisticshowto.com/v-statistics
    """
    # Ensure df is a pandas DataFrame
    df = to_pd_df(df)

    # Get dtypes for all columns
    col_dtypes = [get_column_dtype(df[col]) for col in df.columns]

    # Selecting categorical variables
    categorical_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "categorical"]
    ].columns.tolist()
    categorical_vars = df[categorical_cols]

    # Initialize an empty DataFrame to store correlation values
    corr_matrix = pd.DataFrame(
        index=categorical_vars.columns, columns=categorical_vars.columns
    )

    # Calculate Cramér's V for each pair of categorical variables
    for col1 in categorical_vars.columns:
        for col2 in categorical_vars.columns:
            if col1 != col2:
                contingency_table = pd.crosstab(
                    categorical_vars[col1], categorical_vars[col2]
                )
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                n = np.sum(contingency_table.values)
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                corr_matrix.loc[col1, col2] = cramers_v

    # Filling diagonal with 1s (as a variable is perfectly correlated with itself)
    np.fill_diagonal(corr_matrix.values, 1)

    return corr_matrix


def calc_categorical_categorical_corr_series(
    s1: Union[pd.Series, pl.Series], s2: Union[pd.Series, pl.Series]
) -> float:
    r"""
    Calculate the correlation (Cramér's V) between two categorical variables.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The first categorical variable.
    s2 : Union[pd.Series, pl.Series]
        The second categorical variable.

    Returns
    -------
    float
        The correlation between the two categorical variables.

    Notes
    -----
    Cramér's :math:`V` is a measure of association between two categorical variables
    and ranges from :math:`0` to :math:`+1` (inclusive).

    Cramér's :math:`V` is defined as the square root of the chi-squared statistic
    divided by the sample size:

    .. math::

        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`\chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    It is also known as the mean square contingency coefficient. It is a
    generalization of the phi coefficient, which is used when both variables are
    binary. In particular, in the case of a 2x2 contingency table, Cramér's :math:`V`
    is equivalent to the absolute value of the phi coefficient.

    Cramér's :math:`V` is symmetric. That is, the correlation between variable
    :math:`A` and variable :math:`B` is the same as the correlation between variable
    :math:`B` and variable :math:`A`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    - https://www.statisticshowto.com/v-statistics
    """
    # Ensure s1 and s2 are pandas Series
    s1 = to_pd_s(s1)
    s2 = to_pd_s(s2)

    # make sure both series are categorical
    if get_column_dtype(s1) != "categorical":
        raise TypeError(f"s1 is `{get_column_dtype(s1)}`, not categorical")
    if get_column_dtype(s2) != "categorical":
        raise TypeError(f"s2 is `{get_column_dtype(s2)}`, not categorical")

    contingency_table = pd.crosstab(s1, s2)
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table.values)
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
