import pytest

from predictables.univariate.src.plots.util._get_significance_band import (
    get_significance_band,
)


@pytest.mark.parametrize(
    "p_value, expected",
    [
        # Clear-cut cases
        (0.005, "Extremely likely that test_statistic is significant"),
        (0.025, "Very likely that test_statistic is significant"),
        (0.075, "Somewhat likely that test_statistic is significant"),
        (0.15, "Unlikely that test_statistic is significant"),
        (0.5, "Unlikely that test_statistic is significant"),
        # Edge cases: boundary values
        (0.0, "Extremely likely that test_statistic is significant"),
        (0.01, "Very likely that test_statistic is significant"),
        (0.05, "Somewhat likely that test_statistic is significant"),
        (0.10, "Unlikely that test_statistic is significant"),
    ],
)
def test_get_significance_band(p_value, expected):
    """
    Test the get_significance_band function with various p_values to ensure
    it returns the correct significance statement.
    """
    # We'll use a generic statistic name as the function's behavior does not depend on it
    statistic = "test_statistic"
    assert get_significance_band(p_value, statistic) == expected


# Negative p_value should raise an error
@pytest.mark.parametrize("p_value", [(-0.01,), (-0.05,), (-0.10,)])
def test_get_significance_band_negative_p_value(p_value):
    """
    Test the get_significance_band function with a negative p_value to ensure
    it handles or rejects such input appropriately.
    """
    statistic = "test_statistic"
    with pytest.raises(ValueError) as err:
        get_significance_band(p_value[0], statistic)
    assert str(err.value) == "p_value must be non-negative", (
        "Expected ValueError with message 'p_value must be non-negative', "
        f"but got {err.value}"
    )
