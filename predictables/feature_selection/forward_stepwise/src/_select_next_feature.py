from typing import List, Optional

import pandas as pd
from statsmodels.formula.api import ols


# Function to select the next feature
def select_next_feature(
    data: pd.DataFrame,
    selected_features: List[str],
    remaining_features: List[str],
    target: str,
    p_value_test: float = 0.05,
) -> Optional[str]:
    best_p_value = 1.0
    best_feature = None

    for feature in remaining_features:
        formula = f"{target} ~ " + " + ".join(selected_features + [feature])
        model = ols(formula, data=data).fit()

        p_value = model.pvalues[feature]
        if p_value < p_value_test and p_value < best_p_value:
            best_p_value = p_value
            best_feature = feature

    return best_feature
