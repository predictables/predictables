from typing import List

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def fit_model(
    data: pd.DataFrame, features: List[str], target: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    formula = f"{target} ~ " + " + ".join(features)
    model = ols(formula, data=data).fit()
    return model
