from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm  # type: ignore
from scipy import stats  # type: ignore

from predictables.util import (
    get_column_dtype,
    select_cols_by_dtype,
    to_pd_df,
    to_pd_s,
)


def calc_continuous_categorical_corr(
    *args: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    method: str = "spearman",
) -> Union[float, pd.DataFrame]:
    """
    Calculates the correlation either between a continuous and categorical variable or
    between all pairs of continuous and categorical variables in a data frame.

    Parameters
    ----------
    method : str
        The method to use to calculate the correlation. Must be one of the following:
        - "spearman"
        - "anova"
        - "ancova"
    *args : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The arguments to pass to the appropriate function.

        - If one argument is passed, it is assumed to be a data
          frame or some other similar data structure.
        - If two arguments are passed, they are assumed to be a continuous
          variable and a categorical variable.

    Returns
    -------
    Union[float, pd.DataFrame]
        The correlation between a continuous and categorical variable or between all
        pairs of continuous and categorical variables in a data frame.

    Notes
    -----
    The correlation between a continuous variable and a categorical variable is
    calculated using one of three methods:

    - Spearman's rank correlation
    - ANOVA
    - ANCOVA

    The Spearman's rank correlation is used for binary categorical variables and
    multi-level categorical variables. The ANOVA and ANCOVA methods are used for
    binary categorical variables and multi-level categorical variables, respectively.
    """
    if len(args) == 1:
        return calc_continuous_categorical_corr_df(args[0], method)  # type: ignore
    elif len(args) == 2:
        return calc_continuous_categorical_corr_series(args[0], args[1], method)  # type: ignore
    else:
        raise TypeError(
            f"Invalid number of arguments: Must be 1 or 2, but got {len(args)}"
        )


def calc_continuous_categorical_corr_df(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Calculates the correlation between all pairs of continuous and categorical
    variables in a data frame.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataset to analyze.
    method : str
        The method to use to calculate the correlation. Must be one of the following:
        - "spearman"
        - "anova"
        - "ancova"

    Returns
    -------
    pd.DataFrame
        A dataframe containing the correlation between all pairs of continuous and
        categorical variables in the dataset.

    Notes
    -----
    The correlation between a continuous variable and a categorical variable is
    calculated using one of three methods:

    - Spearman's rank correlation
    - ANOVA
    - ANCOVA

    The Spearman's rank correlation is used for binary categorical variables and
    multi-level categorical variables. The ANOVA and ANCOVA methods are used for
    binary categorical variables and multi-level categorical variables, respectively.
    """
    # Ensure that the input is a DataFrame
    df = to_pd_df(df)

    # Ensure that the method is valid
    method = method.lower()
    if method not in ["spearman", "anova", "ancova"]:
        raise ValueError(
            "method must be one of the following: 'spearman', 'anova', 'ancova', but "
            f"got {method}"
        )

    # Calculate the correlation between continuous and categorical variables
    if method == "spearman":
        return spearmans_rank_correlation_coef_df(df)
    elif method == "anova":
        return anova_correlation_coef_df(df)
    elif method == "ancova":
        return ancova_correlation_coef_df(df)
    else:
        raise ValueError(
            "method must be one of the following: 'spearman', 'anova', 'ancova', but "
            f"got {method}"
        )


def calc_continuous_categorical_corr_series(
    s1: Union[pd.Series, pl.Series],
    s2: Union[pd.Series, pl.Series],
    method: str = "spearman",
) -> float:
    """
    Calculates the correlation between a continuous and categorical variable.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The continuous variable.
    s2 : Union[pd.Series, pl.Series]
        The categorical variable.
    method : str
        The method to use to calculate the correlation. Must be one of the following:
        - "spearman"
        - "anova"
        - "ancova"

    Returns
    -------
    float
        The correlation between the continuous and categorical variables.

    Notes
    -----
    The correlation between a continuous variable and a categorical variable is
    calculated using one of three methods:

    - Spearman's rank correlation
    - ANOVA
    - ANCOVA

    The Spearman's rank correlation is used for binary categorical variables and
    multi-level categorical variables. The ANOVA and ANCOVA methods are used for
    binary categorical variables and multi-level categorical variables, respectively.
    """
    # Ensure that the method is valid
    method = method.lower()
    if method not in ["spearman", "anova", "ancova"]:
        raise ValueError(
            "method must be one of the following: 'spearman', 'anova', 'ancova', "
            f"but got {method}"
        )

    # Calculate the correlation between continuous and categorical variables
    if method == "spearman":
        return spearmans_rank_correlation_coef_series(s1, s2)
    elif method == "anova":
        return anova_correlation_coef_series(s1, s2)
    elif method == "ancova":
        return ancova_correlation_coef_series(s1, s2)
    else:
        raise ValueError(
            "method must be one of the following: 'spearman', 'anova', 'ancova', "
            f"but got {method}"
        )


def anova_correlation_coef_df(
    df: Union[pd.DataFrame, pl.DataFrame],
) -> pd.DataFrame:
    r"""
    Calculate the correlation between continuous and categorical variables in a df.
    This function uses ANOVA to compute the correlation.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        The data frame containing continuous and categorical variables.

    Returns
    -------
    pd.DataFrame
        A data frame containing the correlation between continuous and categorical
        variables.

    Notes
    -----
    Eta Squared (η²) is used to quantify the effect size in the context of ANOVA
    when assessing the relationship between a continuous variable and a categorical
    variable.

    ### Calculation

    η² is calculated as the ratio of the 'sum of squares between groups' (SS_between) to
    the 'total sum of squares' (SS_total):

    $$
    η² = \frac{SS_{between}}{SS_{total}}
    $$

    ### Interpretation

    η² represents the proportion of the variance in the continuous variable that is
    attributable to the categorical variable. It ranges from 0 to 1, where 0 indicates
    no variance explained by the categorical variable (no effect), and 1 indicates
    complete variance explanation (maximum effect).

    ### Remark

    η² is a measure of effect size, not a direct correlation coefficient. It should
    not be interpreted as a measure of linear correlation like Pearson's r. The square
    root of η² does not represent a correlation coefficient and should be used
    cautiously. η² is valuable in quantifying the relative importance of factors in
    ANOVA.

    References
    ----------
    - https://en.wikipedia.org/wiki/Analysis_of_variance
    - https://www.statisticshowto.com/probability-and-statistics/f-statistic-value-anova/
    """
    # Separate continuous and categorical variables
    continuous_vars = select_cols_by_dtype(df, "continuous").columns
    categorical_vars = select_cols_by_dtype(df, "categorical").columns

    # Initialize an empty DataFrame to store correlation values
    corr_matrix = pd.DataFrame(index=continuous_vars, columns=categorical_vars)

    # Calculate ANOVA for each pair of continuous and categorical variables
    for cont_col in continuous_vars:
        for cat_col in categorical_vars:
            groups = [
                pd.Series(group[cont_col]).dropna()
                for _, group in list(df.groupby(cat_col))
            ]
            _ = len(groups)
            _ = sum(len(g) for g in groups)
            ss_between = sum(
                len(g) * (np.mean(g) - np.mean(df[cont_col])) ** 2 for g in groups
            )
            ss_total = sum((val - np.mean(df[cont_col])) ** 2 for val in df[cont_col])
            eta_sq = ss_between / ss_total

            corr_matrix.loc[cont_col, cat_col] = eta_sq

    return corr_matrix


def anova_correlation_coef_series(
    s1: Union[pd.Series, pl.Series],
    s2: Union[pd.Series, pl.Series],
) -> float:
    r"""
    Calculate the correlation between a continuous and categorical variable.
    This function uses ANOVA to compute the eta squared correlation.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The continuous variable.
    s2 : Union[pd.Series, pl.Series]
        The categorical variable.

    Returns
    -------
    float
        The correlation between the continuous and categorical variables.

    Notes
    -----
    Eta Squared (η²) is used to quantify the effect size in the context of ANOVA
    when assessing the relationship between a continuous variable and a categorical
    variable.

    ### Calculation

    :math:`\eta^2` is calculated as the ratio of the 'sum of squares between groups'
    :math:`(SS_{\text{between}})` to the 'total sum of squares'
    :math:`(SS_{\text{total}})`:

    .. math::
        \eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}

    ### Interpretation

    :math:`\eta^2` represents the proportion of the variance in the continuous
    variable that is attributable to the categorical variable. It ranges from
    :math:`0` to :math:`1`, where :math:`0` indicates no variance explained
    by the categorical variable (no effect), and :math:`1` indicates complete
    variance explanation (maximum effect).

    ### Remark

    :math:`\eta^2` is a measure of effect size, not a direct correlation coefficient.
    It should not be interpreted as a measure of linear correlation like Pearson's
    :math:`r`. The square root of :math:`\eta^2` does not represent a correlation
    coefficient and should be used cautiously.

    :math:`\eta^2` is valuable in quantifying the relative importance of factors in
    ANOVA, given these assumptions:

    - The data are at least approximately normally distributed.
    - The groups have approximately equal sample sizes.
    - The groups have approximately equal variance.

    References
    ----------
    - https://en.wikipedia.org/wiki/Analysis_of_variance
    - https://www.statisticshowto.com/probability-and-statistics/f-statistic-value-anova/
    """
    # Make sure the series are pandas series
    s1, s2 = to_pd_s(s1), to_pd_s(s2)

    # Ensure that one series is continuous and the other is categorical
    s1_type = get_column_dtype(s1)
    s2_type = get_column_dtype(s2)

    if s1_type == "continuous" and s2_type == "categorical":
        pass
    elif s1_type == "categorical" and s2_type == "continuous":
        s1, s2 = s2, s1
    else:
        raise TypeError(
            f"Expected one continuous and one categorical variable, but got {s1_type} "
            f"and {s2_type}"
        )

    # Calculate ANOVA
    groups = [s1[s2 == val] for val in s2.unique()]
    # Calculate Eta Squared
    _ = len(groups)
    _ = sum(len(g) for g in groups)
    ss_between = sum(len(g) * (np.mean(g) - np.mean(s1)) ** 2 for g in groups)
    ss_total = sum((val - np.mean(s1)) ** 2 for val in s1)
    return ss_between / ss_total


def ancova_correlation_coef_df(
    df: Union[pd.DataFrame, pl.DataFrame],
) -> pd.DataFrame:
    r"""
    Calculate the correlation between continuous and categorical variables in a df.
    This function uses ANCOVA to compute the correlation.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        The data frame containing continuous and categorical variables.

    Returns
    -------
    pd.DataFrame
        A data frame containing the correlation between continuous and categorical
        variables.

    Notes
    -----
    Partial Eta Squared :math:`(\eta^2_{\text{partial}})` is used to assess the effect
    size in ANCOVA, accounting for the influence of covariates.

    ### Calculation

    :math:`\eta^2_{\text{partial}}` is calculated as the ratio of the 'sum of squares
    due to the factor' :math:`(SS_{\text{factor}})` to the sum of
    :math:`SS_{\text{factor}}` and the 'error sum of squares'
    :math:`(SS_{\text{error}})`:

    .. math::

        \eta^2_{\text{partial}} = \frac{SS_{\text{factor}}}{SS_{\text{factor}} +
        SS_{\text{error}}}

    ### Interpretation

    :math:`\eta^2_{\text{partial}}` indicates the proportion of the total variance in
    the continuous variable that is attributable to a categorical variable, after
    controlling for covariates. Its values range from :math:`0` to :math:`1`. A value
    of :math:`0` suggests no explanatory power of the categorical variable over and
    above the covariates, whereas a value of :math:`1` indicates that the categorical
    variable accounts for all the variance in the continuous variable, beyond what
    is already explained by the covariates.

    ### Remark

    Similar to :math:`\eta^2`, :math:`\eta^2_{\text{partial}}` is not a correlation
    coefficient and does not imply a linear relationship. It quantifies the relative
    importance of a categorical variable in the presence of other covariates.
    The square root of :math:`\eta^2_{\text{partial}}` should not be interpreted as
    a correlation coefficient. This measure is particularly useful in multivariate
    contexts where the effects of multiple variables are considered simultaneously.

    References
    ----------
    - https://en.wikipedia.org/wiki/Analysis_of_covariance
    - https://www.statisticshowto.com/ancova/
    """
    # Separate continuous and categorical variables
    continuous_vars = select_cols_by_dtype(df, "continuous").columns
    categorical_vars = select_cols_by_dtype(df, "categorical").columns

    # Initialize an empty DataFrame to store correlation values
    effect_size_matrix = pd.DataFrame(index=continuous_vars, columns=categorical_vars)

    # Calculate ANCOVA for each pair of continuous and categorical variables
    for cont_col in continuous_vars:
        for cat_col in categorical_vars:
            # ANCOVA model formula - the covariates are just the other levels
            # of the categorical variable
            model_formula = f"{cont_col} ~ C({cat_col})"
            model = sm.OLS.from_formula(model_formula, df).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)

            ss_factor = aov_table.loc[f"C({cat_col})", "sum_sq"]
            ss_error = aov_table.loc["Residual", "sum_sq"]
            eta_sq_partial = ss_factor / (ss_factor + ss_error)

            effect_size_matrix.loc[cont_col, cat_col] = eta_sq_partial

    return effect_size_matrix


def ancova_correlation_coef_series(
    s1: Union[pd.Series, pl.Series],
    s2: Union[pd.Series, pl.Series],
) -> float:
    r"""
    Calculate the correlation between a continuous and categorical variable.
    This function uses ANCOVA to compute the correlation.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The continuous variable.
    s2 : Union[pd.Series, pl.Series]
        The categorical variable.

    Returns
    -------
    float
        The correlation between the continuous and categorical variables.

    Notes
    -----
    Partial Eta Squared :math:`(\eta^2_{\text{partial}})` is used to assess
    the effect size in ANCOVA, accounting for the influence of covariates.

    ### Calculation

    :math:`\eta^2_{\text{partial}}` is calculated as the ratio of the 'sum of
    squares due to the factor' :math:`(SS_{\text{factor}})` to the sum of
    :math:`SS_{\text{factor}}` and the 'error sum of squares'
    :math:`(SS_{\text{error}})`:

    .. math::

        \eta^2_{\text{partial}} = \frac{SS_{\text{factor}}}{SS_{\text{factor}} +
        SS_{\text{error}}}

    ### Interpretation

    :math:`\eta^2_{\text{partial}}` indicates the proportion of the total variance in
    the continuous variable that is attributable to a categorical variable, after
    controlling for covariates. Its values range from :math:`0` to :math:`1`. A value
    of :math:`0` suggests no explanatory power of the categorical variable over and
    above the covariates, whereas a value of :math:`1` indicates that the categorical
    variable accounts for all the variance in the continuous variable, beyond what is
    already explained by the covariates.

    ### Remark

    Similar to :math:`\eta^2`, :math:`\eta^2_{\text{partial}}` is not a correlation
    coefficient and does not imply a linear relationship. It quantifies the relative
    importance of a categorical variable in the presence of other covariates.
    The square root of :math:`\eta^2_{\text{partial}}` should not be interpreted
    as a correlation coefficient. This measure is particularly useful in
    multivariate contextswhere the effects of multiple variables are considered
    simultaneously.

    References
    ----------
    - https://en.wikipedia.org/wiki/Analysis_of_covariance
    - https://www.statisticshowto.com/ancova/
    """
    # Make sure the series are pandas series
    s1, s2 = to_pd_s(s1), to_pd_s(s2)

    # Ensure that one series is continuous and the other is categorical
    s1_type = get_column_dtype(s1)
    s2_type = get_column_dtype(s2)

    if s1_type == "continuous" and s2_type == "categorical":
        pass
    elif s1_type == "categorical" and s2_type == "continuous":
        s1, s2 = s2, s1
    else:
        raise TypeError(
            "Expected one continuous and one categorical variable, but got "
            f"{s1_type} and {s2_type}"
        )

    # ANCOVA model formula - the covariates are just the other levels of the
    # categorical variable
    df = pd.concat([s1, s2], axis=1)
    df.columns = pd.Index(["cont", "cat"])
    model_formula = "cont ~ C(cat)"
    model = sm.OLS.from_formula(model_formula, df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)

    ss_factor = aov_table.loc["C(cat)", "sum_sq"]
    ss_error = aov_table.loc["Residual", "sum_sq"]
    return ss_factor / (ss_factor + ss_error)


def spearmans_rank_correlation_coef_df(
    df: Union[pd.DataFrame, pl.DataFrame],
) -> pd.DataFrame:
    r"""
    Calculate a rank-based correlation between continuous and categorical variables in
    a df.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        The data frame containing continuous and categorical variables.

    Returns
    -------
    pd.DataFrame
        A data frame containing the correlation coefficients between continuous and
        categorical variables.

    Notes
    -----
    Spearman's Rank Correlation Coefficient is utilized to evaluate the strength and
    direction of the monotonic relationship between a continuous variable and a
    numerically encoded categorical variable.

    ### Calculation

    Spearman's correlation coefficient :math:`\rho` is computed as the Pearson
    correlation coefficient :math:`(r)` between the rank values of the two variables:

    .. math::
        rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}

    where :math:`d_i` is the difference between the ranks of corresponding values,
    and :math:`n` is the number of observations.

    ### Interpretation

    Spearman's :math:`\rho` ranges from :math:`-1` to :math:`+1`. A :math:`rho`
    value of :math:`+1` indicates a perfect positive monotonic relationship,
    :math:`-1` indicates a perfect negative monotonic relationship, and :math:`0`
    implies no monotonic relationship. It assesses how well the relationship between
    two variables can be described using a monotonic function.

    ### Applicability

    This non-parametric measure is suitable when the data do not meet the assumptions
    of normality required for Pearson's correlation. It is especially useful in
    exploratory data analysis to identify potential monotonic relationships without
    making assumptions about the linearity of the relationship.

    ### Remark

    For categorical variables with more than two levels, a numerical encoding that
    preserves the order (if any) of the categories is essential. Spearman's rho should
    be interpreted with the nature of the categorical variable in mind, as it reflects
    the strength of a monotonic, not necessarily linear, relationship.

    References
    ----------
    - https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    - https://www.statisticshowto.com/spearman-rank-correlation-definition-calculate/
    """
    # Ensure that the input is a DataFrame
    df = to_pd_df(df)

    # Separate continuous and categorical variables
    continuous_vars = select_cols_by_dtype(df, "continuous").columns
    categorical_vars = select_cols_by_dtype(df, "categorical").columns

    # Initialize an empty DataFrame to store correlation values
    corr_matrix = pd.DataFrame(
        index=continuous_vars, columns=categorical_vars, dtype=float
    )

    # Calculate rank-based correlations for each pair of continuous and
    # categorical variables
    for cont_col in continuous_vars:
        for cat_col in categorical_vars:
            if df[cat_col].nunique() == 2:  # Binary categorical variable
                # Spearman's rank correlation for binary categorical variables
                corr, _ = stats.spearmanr(
                    df[cont_col], df[cat_col].astype("category").cat.codes
                )
            else:  # Multi-level categorical variable
                # Rank the continuous variable
                _ = stats.rankdata(df[cont_col])
                # Group by categorical variable and calculate mean rank
                grouped_ranks = df.groupby(cat_col, observed=True)[cont_col].mean()
                # Calculate correlation between categorical variable levels
                # and mean ranks
                corr, _ = stats.spearmanr(grouped_ranks.index, grouped_ranks)

            corr_matrix.at[cont_col, cat_col] = corr

    return corr_matrix


def spearmans_rank_correlation_coef_series(
    s1: Union[pd.Series, pl.Series],
    s2: Union[pd.Series, pl.Series],
) -> float:
    r"""
    Calculate a rank-based correlation between a continuous and categorical variable.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The continuous variable.
    s2 : Union[pd.Series, pl.Series]
        The categorical variable.

    Returns
    -------
    float
        The correlation between the continuous and categorical variables.

    Notes
    -----
    Spearman's Rank Correlation Coefficient is utilized to evaluate the strength and
    direction of the monotonic relationship between a continuous variable and a
    numerically encoded categorical variable.

    ### Calculation

    Spearman's correlation coefficient (rho) is computed as the Pearson correlation
    coefficient between the rank values of the two variables:

    $$
    rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
    $$

    where \(d_i\) is the difference between the ranks of corresponding values, and \(n\)
    is the number of observations.

    ### Interpretation

    Spearman's rho ranges from -1 to +1. A rho value of +1 indicates a perfect
    positive monotonic relationship, -1 indicates a perfect negative monotonic
    relationship, and 0 implies no monotonic relationship. It assesses how well
    the relationship between two variables can be described using a monotonic
    function.

    ### Applicability

    This non-parametric measure is suitable when the data do not meet the
    assumptions of normality required for Pearson's correlation. It is especially useful
    in exploratory data analysis to identify potential monotonic relationships without
    making assumptions about the linearity of the relationship.

    ### Remark

    For categorical variables with more than two levels, a numerical encoding that
    preserves the order (if any) of the categories is essential. Spearman's rho
    should be interpreted with the nature of the categorical variable in mind,
    as it reflects the strength of a monotonic, not necessarily linear, relationship.

    References
    ----------
    - https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    - https://www.statisticshowto.com/spearman-rank-correlation-definition-calculate/
    """
    # Make sure the series are pandas series
    s1, s2 = to_pd_s(s1), to_pd_s(s2)

    # Ensure that one series is continuous and the other is categorical
    s1_type = get_column_dtype(s1)
    s2_type = get_column_dtype(s2)

    if s1_type == "continuous" and s2_type == "categorical":
        pass
    elif s1_type == "categorical" and s2_type == "continuous":
        s1, s2 = s2, s1
    else:
        raise TypeError(
            "Expected one continuous and one categorical variable, but got "
            f"{s1_type} and {s2_type}"
        )

    # Calculate rank-based correlation differently depending on whether
    # the categorical variable is binary or multi-level
    if s2.nunique() == 2:
        # Spearman's rank correlation for binary categorical variables
        corr, _ = stats.spearmanr(s1, s2.astype("category").cat.codes)
    else:
        # Calculate correlation between categorical variable levels and mean ranks
        _ = stats.rankdata(s1)
        grouped_ranks = s1.groupby(s2, observed=True).mean()
        corr, _ = stats.spearmanr(grouped_ranks.index, grouped_ranks)

    return corr
