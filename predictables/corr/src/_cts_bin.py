from typing import Union

import pandas as pd
import polars as pl

from predictables.util import get_column_dtype, to_pd_df, to_pd_s


def calc_continuous_binary_corr(
    *args: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> Union[float, pd.DataFrame]:
    r"""
    Calculate the point-biserial correlation coefficient either between two
    continuous-binary variables or between all pairs of continuous-binary variables
    in a data frame.

    Parameters
    ----------
    *args : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The arguments to pass to the appropriate function.

        - If one argument is passed, it is assumed to be a data
          frame or some other similar data structure.
        - If two arguments are passed, they are assumed to be two
          continuous-binary variables.

    Returns
    -------
    Union[float, pd.DataFrame]
        The correlation between two continuous-binary variables or between all pairs of
        continuous-binary variables in a data frame.

    Notes
    -----
    The point-biserial correlation coefficient is a measure of association between a
    continuous variable and a binary variable. It is defined as the Pearson correlation
    coefficient between the continuous variable and a dummy variable representing the
    binary variable:

    .. math::

        r_{pb} = \frac{\bar{x}_1 -\bar{x}_0}{s} \sqrt{\frac{n_1 n_0}{n^2}}


    where :math:`\bar{x}_1` and :math:`\bar{x}_0` are the means of the continuous
    variable for the two groups defined by the binary variable, :math:`s` is the
    standard deviation of the continuous variable, :math:`n_1` and :math:`n_0` are
    the number of observations in the two groups, and :math:`n = n_0 + n_1` is the
    total number of observations.

    References
    ----------
    - https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    - https://www.statisticshowto.com/point-biserial-correlation/
    """
    if len(args) == 1:
        if isinstance(args[0], (pd.Series, pl.Series)):
            return calc_continuous_binary_corr_df(to_pd_df(args[0].to_frame()))
        if isinstance(args[0], (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
            return calc_continuous_binary_corr_df(args[0])
        else:
            raise TypeError(
                "Invalid argument type: Expected pandas.core.frame.DataFrame, "
                "polars.dataframe.frame.DataFrame, or LazyFrame"
            )
    elif len(args) == 2:
        if isinstance(args[0], (pd.Series, pl.Series)) and isinstance(
            args[1], (pd.Series, pl.Series)
        ):
            return calc_continuous_binary_corr_series(args[0], args[1])
        else:
            raise TypeError(
                "Invalid argument types: Expected pandas.core.series.Series "
                "or polars.series.series.Series"
            )
    else:
        raise TypeError(
            f"Invalid number of arguments: Must be 1 or 2, but got {len(args)}"
        )


def calc_continuous_binary_corr_df(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> pd.DataFrame:
    r"""
    Calculate the point-biserial correlation coefficient for continuous-binary
    variable pairs in a df.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataset to analyze.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the correlation between all pairs of continuous-binary
        variables in the dataset.

    Notes
    -----
    The point-biserial correlation coefficient is a measure of association between a
    continuous variable and a binary variable. It is defined as the Pearson correlation
    coefficient between the continuous variable and a dummy variable representing the
    binary variable:

    .. math::

        r_{pb} = \frac{\bar{x}_1 - \bar{x}_0}{s} \sqrt{\frac{n_1 n_0}{n^2}}

    where :math:`\bar{x}_1` and :math:`\bar{x}_0` are the means of the continuous
    variable for the two groups defined by the binary variable, :math:`s` is the
    standard deviation of the continuous variable, :math:`n_1` and :math:`n_0` are
    the number of observations in the two groups, and :math:`n` is the total number
    of observations.

    References
    ----------
    - https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    - https://www.statisticshowto.com/point-biserial-correlation/
    """
    # Ensure that the input is a DataFrame
    df = to_pd_df(df)

    # Get dtypes for all columns
    col_dtypes = [get_column_dtype(df[col]) for col in df.columns]

    # Separate continuous and binary variables
    continuous_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "continuous"]
    ].columns.tolist()
    binary_cols = df[
        [df.columns.tolist()[i] for i, x in enumerate(col_dtypes) if x == "binary"]
    ].columns.tolist()

    # Create a df with only continuous/binary variables
    continuous_vars = df[continuous_cols]
    binary_vars = df[binary_cols]

    # Initialize an empty DataFrame to store correlation values
    corr_matrix = pd.DataFrame(
        index=continuous_vars.columns, columns=binary_vars.columns
    )

    # Calculate point-biserial correlation for each pair of continuous and binary
    # variables
    for cont_col in continuous_vars:
        for bin_col in binary_vars:
            # recode binary variable to 0.0/1.0
            lookup = binary_vars[bin_col].drop_duplicates().sort_values().tolist()
            binary_vars.loc[:, [bin_col]] = (
                binary_vars.loc[:, [bin_col]]
                .replace(dict(zip(lookup, [0.0, 1.0])))
                .values.tolist()
            )

            corr_matrix.loc[pd.Index([cont_col]), pd.Index([bin_col])] = (
                continuous_vars[cont_col].corr(binary_vars[bin_col], method="pearson")
            )

    return corr_matrix


def calc_continuous_binary_corr_series(
    s1: Union[pd.Series, pl.Series], s2: Union[pd.Series, pl.Series]
) -> float:
    r"""
    Calculate the point-biserial correlation coefficient for a continuous-binary
    variable pair.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The continuous variable.
    s2 : Union[pd.Series, pl.Series]
        The binary variable.

    Returns
    -------
    float
        The correlation between the continuous and binary variables.

    Notes
    -----
    The point-biserial correlation coefficient is a measure of association between a
    continuous variable and a binary variable. It is defined as the Pearson correlation
    coefficient between the continuous variable and a dummy variable representing the
    binary variable:

    .. math::

        r_{pb} = \frac{\bar{x}_1 - \bar{x}_0}{s} \sqrt{\frac{n_1 n_0}{n^2}}

    where :math:`\bar{x}_1` and :math:`\bar{x}_0` are the means of the continuous
    variable for the two groups defined by the binary variable, :math:`s` is the
    standard deviation of the continuous variable, :math:`n_1` and :math:`n_0` are
    the number of observations in the two groups, and :math:`n` is the total number
    of observations.

    References
    ----------
    - https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    - https://www.statisticshowto.com/point-biserial-correlation/
    """
    # Convert to pandas series
    s1 = to_pd_s(s1)
    s2 = to_pd_s(s2)

    # Ensure that exactly one of the two series is binary, and the other is continuous
    cond1 = (get_column_dtype(s1) == "binary") | (get_column_dtype(s2) == "binary")
    cond2 = (get_column_dtype(s1) == "continuous") | (
        get_column_dtype(s2) == "continuous"
    )
    cond3 = (
        (get_column_dtype(s1) == "binary") & (get_column_dtype(s2) == "continuous")
    ) | ((get_column_dtype(s1) == "continuous") & (get_column_dtype(s2) == "binary"))
    if not (cond1 & cond2 & cond3):
        raise TypeError(
            f"Exactly one of the two series must be binary, and the other must be "
            f"continuous, but s1 is `{get_column_dtype(s1)}` and s2 is "
            f"`{get_column_dtype(s2)}`"
        )

    # Recode binary variable to 0.0/1.0
    if get_column_dtype(s1) == "binary":
        lookup = s1.drop_duplicates().sort_values().tolist()
        s1 = s1.copy()
        s1 = s1.replace(dict(zip(lookup, [0.0, 1.0])))
    else:
        lookup = s2.drop_duplicates().sort_values().tolist()
        s2 = s2.copy()
        s2 = s2.replace(dict(zip(lookup, [0.0, 1.0])))

    # Calculate point-biserial correlation
    return s1.corr(s2, method="pearson")
