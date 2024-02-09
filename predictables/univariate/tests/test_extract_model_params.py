import numpy as np
from scipy.stats import norm  # type: ignore
import polars as pl
import pytest


@pytest.fixture
def logistic_coef():
    return 0.01


@pytest.fixture
def linear_coef():
    return 0.5


@pytest.fixture
def noise():
    np.random.seed(0)
    return norm.rvs(size=500)


@pytest.fixture
def df(noise, logistic_coef, linear_coef):
    return (
        pl.DataFrame({"X0": range(500)})
        # .lazy()
        .with_columns(
            [
                pl.Series(noise).alias("epsilon"),
                pl.col("X0") + pl.Series(noise).alias("X"),
            ]
        )
        # Create linear relationship (easy to predict)
        .with_columns(
            [
                pl.col("X").mul(linear_coef).alias("y_linear"),
            ]
        )
        # Create logistic relationship (harder to predict)
        .with_columns(
            [
                # Logits
                pl.col("X").mul(logistic_coef).alias("logit"),
                # Probability
                pl.col("logit").exp().truediv(1 + pl.col("logit").exp()).alias("prob"),
                # Binary outcome
                pl.when(pl.col("prob") > 0.5).then(1).otherwise(0).alias("y_logistic"),
            ]
        )
    )


@pytest.fixture
def sm_logistic_model(df):
    from predictables.univariate.src._fit_sm_logistic_regression import (
        fit_sm_logistic_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sm_logistic_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_logistic"))
    )


@pytest.fixture
def sm_linear_model(df):
    from predictables.univariate.src._fit_sm_linear_regression import (
        fit_sm_linear_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sm_linear_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_linear"))
    )


@pytest.fixture
def sk_logistic_model(df):
    from predictables.univariate.src._fit_sk_logistic_regression import (
        fit_sk_logistic_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sk_logistic_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_logistic"))
    )


@pytest.fixture
def sk_linear_model(df):
    from predictables.univariate.src._fit_sk_linear_regression import (
        fit_sk_linear_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sk_linear_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_linear"))
    )


def is_close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_extract_model_params_sm_OLS(sm_linear_model, linear_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sm_OLS,
    )

    result = extract_model_params_sm_OLS(sm_linear_model)
    assert is_close(
        result.coef, linear_coef
    ), f"Expected fitted coeficient close to {linear_coef}, got {result.coef}"
    assert is_close(
        result.pvalues, sm_linear_model.pvalues
    ), f"Expected pvalues close to {sm_linear_model.pvalues}, got {result.pvalues}"
    assert is_close(
        result.aic, sm_linear_model.aic
    ), f"Expected AIC close to {sm_linear_model.aic}, got {result.aic}"
    assert is_close(
        result.se, sm_linear_model.bse
    ), f"Expected SE close to {sm_linear_model.bse}, got {result.se}"
    assert is_close(
        result.lower_ci, sm_linear_model.conf_int()[0]
    ), f"Expected lower confidence interval close to {sm_linear_model.conf_int()[0]}, got {result.lower_ci}"
    assert is_close(
        result.upper_ci, sm_linear_model.conf_int()[1]
    ), f"Expected upper confidence interval close to {sm_linear_model.conf_int()[1]}, got {result.upper_ci}"
    assert (
        result.nobs == sm_linear_model.nobs
    ), f"Expected number of observations close to {sm_linear_model.nobs}, got {result.nobs}"
    assert (
        result.df_model == sm_linear_model.df_model
    ), f"Expected model degrees of freedom close to {sm_linear_model.df_model}, got {result.df_model}"


def test_extract_model_params_sm_LogisticRegression(sm_logistic_model, logistic_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sm_LogisticRegression,
    )

    result = extract_model_params_sm_LogisticRegression(sm_logistic_model)
    assert is_close(
        result.coef, logistic_coef
    ), f"Expected fitted coeficient close to {logistic_coef}, got {result.coef}"
    assert is_close(
        result.pvalues, sm_logistic_model.pvalues
    ), f"Expected pvalues close to {sm_logistic_model.pvalues}, got {result.pvalues}"
    assert is_close(
        result.aic, sm_logistic_model.aic
    ), f"Expected AIC close to {sm_logistic_model.aic}, got {result.aic}"
    assert is_close(
        result.se, sm_logistic_model.bse
    ), f"Expected SE close to {sm_logistic_model.bse}, got {result.se}"
    assert is_close(
        result.lower_ci, sm_logistic_model.conf_int()[0]
    ), f"Expected lower confidence interval close to {sm_logistic_model.conf_int()[0]}, got {result.lower_ci}"
    assert is_close(
        result.upper_ci, sm_logistic_model.conf_int()[1]
    ), f"Expected upper confidence interval close to {sm_logistic_model.conf_int()[1]}, got {result.upper_ci}"
    assert (
        result.nobs == sm_logistic_model.nobs
    ), f"Expected number of observations close to {sm_logistic_model.nobs}, got {result.nobs}"
    assert (
        result.df_model == sm_logistic_model.df_model
    ), f"Expected model degrees of freedom close to {sm_logistic_model.df_model}, got {result.df_model}"


def test_extract_model_params_sk_LogisticRegression(sk_logistic_model, logistic_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sk_LogisticRegression,
    )

    result = extract_model_params_sk_LogisticRegression(sk_logistic_model)
    assert is_close(
        result.coef, logistic_coef
    ), f"Expected fitted coeficient close to {logistic_coef}, got {result.coef}"
    assert is_close(
        result.pvalues, sk_logistic_model.pvalues
    ), f"Expected pvalues close to {sk_logistic_model.pvalues}, got {result.pvalues}"
    assert is_close(
        result.aic, sk_logistic_model.aic
    ), f"Expected AIC close to {sk_logistic_model.aic}, got {result.aic}"
    assert is_close(
        result.se, sk_logistic_model.bse
    ), f"Expected SE close to {sk_logistic_model.bse}, got {result.se}"
    assert is_close(
        result.lower_ci, sk_logistic_model.conf_int()[0]
    ), f"Expected lower confidence interval close to {sk_logistic_model.conf_int()[0]}, got {result.lower_ci}"
    assert is_close(
        result.upper_ci, sk_logistic_model.conf_int()[1]
    ), f"Expected upper confidence interval close to {sk_logistic_model.conf_int()[1]}, got {result.upper_ci}"
    assert (
        result.nobs == sk_logistic_model.nobs
    ), f"Expected number of observations close to {sk_logistic_model.nobs}, got {result.nobs}"
    assert (
        result.df_model == sk_logistic_model.df_model
    ), f"Expected model degrees of freedom close to {sk_logistic_model.df_model}, got {result.df_model}"


def test_extract_model_params_sk_LinearRegression(sk_linear_model, linear_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sk_LinearRegression,
    )

    result = extract_model_params_sk_LinearRegression(sk_linear_model)
    assert is_close(
        result.coef, linear_coef
    ), f"Expected fitted coeficient close to {linear_coef}, got {result.coef}"
    assert is_close(
        result.pvalues, sk_linear_model.pvalues
    ), f"Expected pvalues close to {sk_linear_model.pvalues}, got {result.pvalues}"
    assert is_close(
        result.aic, sk_linear_model.aic
    ), f"Expected AIC close to {sk_linear_model.aic}, got {result.aic}"
    assert is_close(
        result.se, sk_linear_model.bse
    ), f"Expected SE close to {sk_linear_model.bse}, got {result.se}"
    assert is_close(
        result.lower_ci, sk_linear_model.conf_int()[0]
    ), f"Expected lower confidence interval close to {sk_linear_model.conf_int()[0]}, got {result.lower_ci}"
    assert is_close(
        result.upper_ci, sk_linear_model.conf_int()[1]
    ), f"Expected upper confidence interval close to {sk_linear_model.conf_int()[1]}, got {result.upper_ci}"
    assert (
        result.nobs == sk_linear_model.nobs
    ), f"Expected number of observations close to {sk_linear_model.nobs}, got {result.nobs}"
    assert (
        result.df_model == sk_linear_model.df_model
    ), f"Expected model degrees of freedom close to {sk_linear_model.df_model}, got {result.df_model}"
