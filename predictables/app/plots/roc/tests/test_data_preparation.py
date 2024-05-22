import subprocess
import pytest
import pandas as pd
from predictables.app.plots.roc.src.data_preparation import load_data, prepare_data


def test_load_data():
    """Test the load_data function."""
    # create a csv file:
    subprocess.run(
        'echo "feature,target,fold\n1,0,0\n2,1,0\n3,0,1\n4,1,1\n5,0,2\n6,1,2" > sample_data.csv',  # noqa: S603, S607 # this is ok -- just creating a sample file
        check=True,
    )
    subprocess.run(
        'echo "feature,target,cross_validation_fold\n1,0,0\n2,1,0\n3,0,1\n4,1,1\n5,0,2\n6,1,2" > sample_error.csv',  # noqa: S603, S607 # this is ok -- just creating a sample file
        check=True,
    )
    data = load_data("sample_data.csv")
    assert isinstance(data, pd.DataFrame), f"Expected DataFrame, got {type(data)}"
    assert "fold" in data.columns, f"Expected 'fold' in columns, got {data.columns}"

    with pytest.raises(KeyError) as exc_info:
        load_data("sample_error.csv", cross_validation_fold=True)
    assert "cross_validation_fold" in str(
        exc_info.value
    ), f"Expected 'cross_validation_fold' in error message, got {exc_info.value}"

    # clean up
    subprocess.run("rm sample_data.csv", check=True)  # noqa: S603, S607 # this is ok -- removing the sample file
    subprocess.run("rm sample_error.csv", check=True)  # noqa: S603, S607 # this is ok -- removing the sample file


def test_prepare_data_time_series():
    """Test the prepare_data function with time series validation."""
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )
    prepared_data = prepare_data(data, use_time_series_validation=True)
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
    prepared_data = prepare_data(data, use_time_series_validation=False)
    assert (
        len(prepared_data) == 3
    ), f"Expected 3, got len(prepared_data) == {len(prepared_data)}"
    assert all(
        isinstance(pair, tuple) for pair in prepared_data
    ), "Expected all pairs to be tuples"
    assert all(
        len(pair) == 2 for pair in prepared_data
    ), "Expected all pairs to have length 2"
