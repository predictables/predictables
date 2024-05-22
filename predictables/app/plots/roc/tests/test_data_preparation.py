import pytest
import pandas as pd
from predictables.app.plots.roc.src.data_preparation import load_data, prepare_roc_data


def test_load_data():
    """Test the load_data function."""
    data = load_data("predictables/app/plots/roc/tests/sample_data.csv")
    assert isinstance(data, pd.DataFrame), f"Expected DataFrame, got {type(data)}"
    assert "fold" in data.columns, f"Expected 'fold' in columns, got {data.columns}"

    with pytest.raises(KeyError) as exc_info:
        load_data("predictables/app/plots/roc/tests/sample_error.csv")
    assert "not found in the data" in str(
        exc_info.value
    ), f"Expected 'not found in the data' in error message, got {exc_info.value}"


def test_prepare_data_time_series():
    """Test the prepare_data function with time series validation."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )
    prepared_data = prepare_roc_data(data, use_time_series_validation=True)
    assert (
        len(prepared_data) == 2
    ), f"Expected 2, got len(prepared_data) == {len(prepared_data)}"
    assert all(
        isinstance(pair, tuple) for pair in prepared_data
    ), "Expected all pairs to be tuples"
    assert all(
        len(pair) == 2 for pair in prepared_data
    ), "Expected all pairs to have length 2"


def test_prepare_data_normal_cv():
    """Test the prepare_data function with normal cross-validation."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )
    prepared_data = prepare_roc_data(data, use_time_series_validation=False)
    assert (
        len(prepared_data) == 3
    ), f"Expected 3, got len(prepared_data) == {len(prepared_data)}"
    assert all(
        isinstance(pair, tuple) for pair in prepared_data
    ), "Expected all pairs to be tuples"
    assert all(
        len(pair) == 2 for pair in prepared_data
    ), "Expected all pairs to have length 2"
