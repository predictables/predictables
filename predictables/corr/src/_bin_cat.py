from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats  # type: ignore

from predictables.util import get_column_dtype, to_pd_df, to_pd_s


def calc_binary_categorical_corr(
    *args: Union[pd.Series, pd.DataFrame],
) -> Union[float, pd.DataFrame]:
    r"""
    Calculates the correlation either between a binary and a categorical variable
    or between all pairs of binary and categorical variables in a data frame.

    Parameters
    ----------
    *args : Union[pd.Series, pd.DataFrame]
        The arguments to pass to the appropriate function.

        - If one argument is passed, it is assumed to be a data
            frame or some other similar data structure.
        - If two arguments are passed, they are assumed to be a
            binary and a categorical variable.

    Returns
    -------
    Union[float, pd.DataFrame]
        The correlation between a binary and a categorical variable or between all
        pairs of binary and categorical variables in a data frame.

    Notes
    -----
    Cramér's :math:`V` is a measure of association between two categorical variables
    and ranges from :math:`0` to :math:`+1` (inclusive). Note that in `predictables`,
    binary variables are treated as a special case of categorical variables.

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
        return calc_binary_categorical_corr_df(pd.DataFrame(args[0]))
    elif len(args) == 2:
        arg1 = to_pd_s(args[0]) if isinstance(args[0], pl.Series) else args[0]  # type: ignore
        return calc_binary_categorical_corr_series(arg1, args[1])  # type: ignore
    else:
        raise TypeError(
            f"Invalid number of arguments: Must be 1 or 2, but got {len(args)}"
        )


def calc_binary_categorical_corr_df(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> pd.DataFrame:
    r"""
    Calculate the correlation (Cramér's V) between binary and categorical variables in
    a df.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The df containing the binary and categorical variables.

    Returns
    -------
    pd.DataFrame
        A correlation matrix of the binary and categorical variables.

    Notes
    -----
    Cramér's :math:`V` is a measure of association between two categorical variables
    and ranges from :math:`0` to :math:`+1` (inclusive). Note that in `predictables`,
    binary variables are treated as a special case of categorical variables.

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

    # Separate binary and categorical variables
    binary_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "binary"]
    ].columns.tolist()
    binary_vars = df[binary_cols]

    categorical_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "categorical"]
    ].columns.tolist()
    categorical_vars = df[categorical_cols]

    # Initialize an empty DataFrame to store correlation values
    corr_matrix = pd.DataFrame(
        index=binary_vars.columns, columns=categorical_vars.columns
    )

    # Calculate Cramér's V for each pair of binary and categorical variables
    for bin_col in binary_vars:
        for cat_col in categorical_vars:
            if bin_col != cat_col:
                contingency_table = pd.crosstab(
                    binary_vars[bin_col], categorical_vars[cat_col]
                )
                chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
                n = np.sum(contingency_table.values)
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                corr_matrix.loc[str(bin_col), str(cat_col)] = cramers_v

    return corr_matrix


def calc_binary_categorical_corr_series(
    s1: Union[pd.Series, pl.Series], s2: Union[pd.Series, pl.Series]
) -> float:
    r"""
    Calculate the correlation (Cramér's V) between a binary and a categorical variable.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The first variable.
    s2 : Union[pd.Series, pl.Series]
        The second variable.

    Returns
    -------
    float
        The correlation between the two variables.

    Notes
    -----
    Cramér's :math:`V` is a measure of association between two categorical variables
    and ranges from :math:`0` to `+1` (inclusive). Note that in PredicTables, binary
    variables are treated as a special case of categorical variables.

    Cramér's :math:`V` is defined as the square root of the chi-squared statistic
    divided by the sample size:

    .. math::
        \phi = \sqrt{\frac{\chi^2}{n}}

    where :math:`\chi^2` is the chi-squared statistic and :math:`n` is the sample size.

    It is also known as the mean square contingency coefficient. It is a generalization
    of the phi coefficient, which is used when both variables are binary. In particular,
    in the case of a 2x2 contingency table, Cramér's :math:`V` is equivalent to the
    absolute value of the phi coefficient.

    Cramér's :math:`V` is symmetric. That is, the correlation between variable :math:`A
    and variable :math:`B` is the same as the correlation between variable :math:`B` and
    variable :math:`A`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    - https://www.statisticshowto.com/v-statistics
    """
    # Ensure s1 and s2 are pandas Series
    s1 = to_pd_s(s1)
    s2 = to_pd_s(s2)

    # Create a df with the two variables
    df = pd.concat([s1, s2], axis=1)

    # Get dtypes for all columns
    col_dtypes = [get_column_dtype(df[col]) for col in df.columns]

    # Separate binary and categorical variables
    binary_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "binary"]
    ].columns.tolist()
    binary_vars = df[binary_cols]

    categorical_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "categorical"]
    ].columns.tolist()
    categorical_vars = df[categorical_cols]

    # Calculate Cramér's V for the pair of binary and categorical variables
    contingency_table = pd.crosstab(
        binary_vars[binary_cols[0]], categorical_vars[categorical_cols[0]]
    )
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table.values)
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
