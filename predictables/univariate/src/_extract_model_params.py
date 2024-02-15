from collections import namedtuple
from typing import Tuple

import statsmodels.api as sm  # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore

from predictables.util import DebugLogger

dbg = DebugLogger(working_file="_extract_model_params.py")

StatsmodelsModelParams = namedtuple(
    "StatsmodelsModelParams",
    [
        "coef",
        "intercept",
        "pvalues",
        "aic",
        "se",
        "lower_ci",
        "upper_ci",
        "n",
        "k",
    ],
)

SKLearnModelParams = namedtuple(
    "SKLearnModelParams",
    ["coef", "k"],
)


def extract_model_params(
    sm_model, sk_model
) -> Tuple[StatsmodelsModelParams, SKLearnModelParams]:
    """
    Extract the parameters from a model and return them as a named tuple. Both the statsmodels and
    sklearn model must be passed, and the function will return the parameters for both models.

    Parameters
    ----------
    sm_model : object
        The fitted statsmodels model
    sk_model : object
        The fitted sklearn model

    Returns
    -------
    Tuple[StatsmodelsModelParams, SKLearnModelParams]
        The model parameters

    """
    dbg.msg("Entering extract_model_params function")
    if isinstance(sm_model, sm.GLM):
        return extract_model_params_sm_GLM(
            sm_model
        ), extract_model_params_sk_LogisticRegression(sk_model)
    elif isinstance(sm_model, sm.OLS):
        return extract_model_params_sm_OLS(
            sm_model
        ), extract_model_params_sk_LinearRegression(sk_model)
    else:
        raise ValueError("Model type not supported")


def extract_model_params_sm_GLM(model: sm.GLM) -> StatsmodelsModelParams:
    """
    Extract the parameters from a statsmodels GLM model and return them as a named tuple.

    Parameters
    ----------
    sm_model : sm.GLM
        The fitted model

    Returns
    -------
    StatsmodelsModelParams
        The model parameters

    """
    dbg.msg("Entering extract_model_params_sm_GLM function")
    return (
        StatsmodelsModelParams(
            model.params.values[1:],
            model.params.values[0],
            model.pvalues,
            model.aic,
            model.bse,
            model.conf_int().values.ravel()[0],
            model.conf_int().values.ravel()[1],
            model.nobs,
            model.df_model,
        )
        if len(model.params) > 1
        else StatsmodelsModelParams(
            model.params.values[0],
            None,
            model.pvalues,
            model.aic,
            model.bse,
            model.conf_int().values.ravel()[0],
            model.conf_int().values.ravel()[1],
            model.nobs,
            model.df_model,
        )
    )


def extract_model_params_sm_OLS(model: sm.OLS) -> StatsmodelsModelParams:
    """
    Extract the parameters from a statsmodels OLS model and return them as a named tuple.

    Parameters
    ----------
    sm_model : sm.OLS
        The fitted model

    Returns
    -------
    StatsmodelsModelParams
        The model parameters

    """
    dbg.msg("Entering extract_model_params_sm_OLS function")
    return (
        StatsmodelsModelParams(
            model.params[1:],
            model.params[0],
            model.pvalues,
            model.aic,
            model.bse,
            model.conf_int().values.ravel()[0],
            model.conf_int().values.ravel()[1],
            model.nobs,
            model.df_model,
        )
        if len(model.params) > 1
        else StatsmodelsModelParams(
            model.params.values[0],
            None,
            model.pvalues,
            model.aic,
            model.bse,
            model.conf_int().values.ravel()[0],
            model.conf_int().values.ravel()[1],
            model.nobs,
            model.df_model,
        )
    )


def extract_model_params_sk_LogisticRegression(
    model: LogisticRegression,
) -> SKLearnModelParams:
    """
    Extract the parameters from a sklearn model and return them as a named tuple.

    Parameters
    ----------
    model : object
        The fitted model

    Returns
    -------
    SKLearnModelParams
        The model parameters

    """
    dbg.msg("Entering extract_model_params_sk function")
    return SKLearnModelParams(
        model.coef_,
        model.coef_.shape[0],
    )


def extract_model_params_sk_LinearRegression(
    model: LinearRegression,
) -> SKLearnModelParams:
    """
    Extract the parameters from a sklearn model and return them as a named tuple.

    Parameters
    ----------
    model : object
        The fitted model

    Returns
    -------
    SKLearnModelParams
        The model parameters

    """
    dbg.msg("Entering extract_model_params_sk function")
    return SKLearnModelParams(
        model.coef_,
        model.coef_.shape[0],
    )
