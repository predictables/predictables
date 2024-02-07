# Import necessary libraries
import pytest
import pandas as pd  # type: ignore
import numpy as np
# from scipy.stats import norm  # type: ignore
# from sklearn.metrics import roc_auc_score  # type: ignore

from predictables.univariate.src.plots._roc_curve_plot import (
    _delong_test_against_chance,
)


@pytest.mark.parametrize(
    "y, yhat, expected_z, expected_p",
    [
        # Test cases will be defined here
    ],
)
def test_delong_test_against_chance(y, yhat, expected_z, expected_p):
    z_stat, p_value = _delong_test_against_chance(pd.Series(y), pd.Series(yhat))
    np.testing.assert_almost_equal(z_stat, expected_z, decimal=5)
    np.testing.assert_almost_equal(p_value, expected_p, decimal=5)


# Additional tests for error handling and edge cases will be defined here
