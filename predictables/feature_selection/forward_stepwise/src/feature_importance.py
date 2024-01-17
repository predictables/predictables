from typing import List

import pandas as pd
import statsmodels.api as sm


def feature_importance(
    model: sm.regression.linear_model.RegressionResultsWrapper, features: List[str]
) -> pd.DataFrame:
    importance = pd.DataFrame({"feature": features, "coefficient": model.params[1:]})
    importance = importance.sort_values("coefficient", ascending=False)
    return importance
