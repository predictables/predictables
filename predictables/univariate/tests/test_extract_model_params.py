import numpy as np
import polars as pl
import pytest
from scipy.stats import norm  # type: ignore


@pytest.fixture
def logistic_coef():
    return 0.003


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
                (pl.col("X0") + pl.Series(noise)).alias("X"),
            ]
        )
        # Create linear relationship (easy to predict)
        .with_columns([pl.col("X").mul(linear_coef).alias("y_linear")])
        # Create logistic relationship (harder to predict)
        .with_columns(
            [
                # Logits
                pl.col("X").mul(logistic_coef).alias("logit")
            ]
        )
        .with_columns(
            [
                # Probability
                pl.col("logit").exp().truediv(1 + pl.col("logit").exp()).alias("prob")
            ]
        )
        .with_columns(
            [
                # Binary outcome
                pl.when(pl.col("prob") > pl.col("prob").median())
                .then(1)
                .otherwise(0)
                .alias("y_logistic")
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
        to_pd_df(df.select("X")), to_pd_s(df.select("y_logistic")["y_logistic"])
    )


@pytest.fixture
def sm_linear_model(df):
    from predictables.univariate.src._fit_sm_linear_regression import (
        fit_sm_linear_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sm_linear_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_linear")["y_linear"])
    )


@pytest.fixture
def sk_logistic_model(df):
    from predictables.univariate.src._fit_sk_logistic_regression import (
        fit_sk_logistic_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sk_logistic_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_logistic")["y_logistic"])
    )


@pytest.fixture
def sk_linear_model(df):
    from predictables.univariate.src._fit_sk_linear_regression import (
        fit_sk_linear_regression,
    )
    from predictables.util import to_pd_df, to_pd_s

    return fit_sk_linear_regression(
        to_pd_df(df.select("X")), to_pd_s(df.select("y_linear")["y_linear"])
    )


def is_close(a, b, tol=1e-3):
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
        result.pvalues.values, sm_linear_model.pvalues.values
    ), f"Expected pvalues close to {sm_linear_model.pvalues}, got {result.pvalues}"
    assert is_close(
        result.aic, sm_linear_model.aic
    ), f"Expected AIC close to {sm_linear_model.aic}, got {result.aic}"
    assert is_close(
        result.se.values, sm_linear_model.bse.values
    ), f"Expected SE close to {sm_linear_model.bse}, got {result.se}"
    assert is_close(
        result.lower_ci, sm_linear_model.conf_int().values.ravel()[0]
    ), f"Expected lower confidence interval close to {sm_linear_model.conf_int().values.ravel()[0]}, got {result.lower_ci}"
    assert is_close(
        result.upper_ci, sm_linear_model.conf_int().values.ravel()[1]
    ), f"Expected upper confidence interval close to {sm_linear_model.conf_int().values.ravel()[1]}, got {result.upper_ci}"
    assert (
        result.n == sm_linear_model.nobs
    ), f"Expected number of observations close to {sm_linear_model.nobs}, got {result.n}"
    assert (
        result.k == sm_linear_model.df_model
    ), f"Expected model degrees of freedom close to {sm_linear_model.df_model}, got {result.k}"


def test_extract_model_params_sm_GLM(sm_logistic_model, logistic_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sm_GLM,
    )

    result = extract_model_params_sm_GLM(sm_logistic_model)
    assert is_close(
        result.coef, logistic_coef
    ), f"Expected fitted coeficient close to {logistic_coef}, got {result.coef}"
    assert is_close(
        result.pvalues.values, sm_logistic_model.pvalues.values
    ), f"Expected pvalues close to {sm_logistic_model.pvalues}, got {result.pvalues}"
    assert is_close(
        result.aic, sm_logistic_model.aic
    ), f"Expected AIC close to {sm_logistic_model.aic}, got {result.aic}"
    assert is_close(
        result.se.values, sm_logistic_model.bse.values
    ), f"Expected SE close to {sm_logistic_model.bse}, got {result.se}"
    assert is_close(
        result.lower_ci, sm_logistic_model.conf_int().values.ravel()[0]
    ), f"Expected lower confidence interval close to {sm_logistic_model.conf_int().values.ravel()[0]}, got {result.lower_ci}"
    assert is_close(
        result.upper_ci, sm_logistic_model.conf_int().values.ravel()[1]
    ), f"Expected upper confidence interval close to {sm_logistic_model.conf_int().values.ravel()[1]}, got {result.upper_ci}"
    assert (
        result.n == sm_logistic_model.nobs
    ), f"Expected number of observations close to {sm_logistic_model.nobs}, got {result.n}"
    assert (
        result.k == sm_logistic_model.df_model
    ), f"Expected model degrees of freedom close to {sm_logistic_model.df_model}, got {result.k}"


def test_extract_model_params_sk_LogisticRegression(sk_logistic_model, logistic_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sk_LogisticRegression,
    )

    result = extract_model_params_sk_LogisticRegression(sk_logistic_model)
    assert is_close(
        result.coef, logistic_coef
    ), f"Expected fitted coeficient close to {logistic_coef}, got {result.coef}"


def test_extract_model_params_sk_LinearRegression(sk_linear_model, linear_coef):
    from predictables.univariate.src._extract_model_params import (
        extract_model_params_sk_LinearRegression,
    )

    result = extract_model_params_sk_LinearRegression(sk_linear_model)
    assert is_close(
        result.coef, linear_coef
    ), f"Expected fitted coeficient close to {linear_coef}, got {result.coef}"
