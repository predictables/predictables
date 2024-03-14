from functools import wraps

import numpy as np
import polars as pl
import pytest

from predictables.util.src._to_pd import to_pd_df
from predictables.util.src._to_pl import to_pl_df
from predictables.util.src._validate_lf import validate_lf


@pytest.fixture
def test_lf():
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()


@pytest.fixture
def test_mapping(test_lf):
    return {
        "pd_df": to_pd_df(test_lf),
        "pl_df": to_pl_df(test_lf),
        "pl_lf": test_lf,
        "np": to_pd_df(test_lf).to_numpy(),
    }


@pytest.mark.parametrize("dtype", ["pd_df", "pl_df", "pl_lf", "np"])
def test_validate_lf(test_lf, test_mapping, dtype):
    lf = test_mapping[dtype]

    @validate_lf
    def test_func(lf):
        return lf

    assert isinstance(
        test_func(lf), pl.LazyFrame
    ), f"Failed for {dtype} - {lf} is not a LazyFrame"


@pytest.mark.parametrize(
    "unsupported_type",
    [[1, 2, 3], 123, 123.0, "string", None, True, False, {"a": 1}, (1, 2, 3)],
)
def test_validate_lf_with_unsupported_type(unsupported_type):
    @validate_lf
    def test_func(lf):
        return lf

    with pytest.raises(ValueError):
        test_func(unsupported_type)


def test_validate_lf_with_multiple_args(test_mapping):
    @validate_lf
    def test_func(lf, column: str):
        return lf.select(column)

    lf = test_mapping["pl_lf"]
    selected_lf = test_func(lf, column="a")

    assert isinstance(selected_lf, pl.LazyFrame), "Result is not a LazyFrame"
    assert "a" in selected_lf.columns, "Column 'a' not found in the LazyFrame"


def test_validate_lf_preserves_data(test_mapping):
    @validate_lf
    def test_func(lf):
        return lf.collect().to_numpy()

    for dtype, lf in test_mapping.items():
        if (
            dtype != "np"
        ):  # numpy arrays don't have column names, so skip direct comparison
            expected_data = (
                to_pl_df(lf).to_numpy() if dtype != "pl_lf" else lf.collect().to_numpy()
            )
            result_data = test_func(lf)
            assert np.array_equal(
                result_data, expected_data
            ), f"Data mismatch for {dtype}"


def test_validate_lf_with_keyword_arg(test_mapping):
    @validate_lf
    def test_func(lf=None):
        return isinstance(lf, pl.LazyFrame)

    for dtype, lf in test_mapping.items():
        assert test_func(
            lf=lf
        ), f"Evaluates to False for {dtype} when passed as keyword arg"
        assert test_func(
            lf
        ), f"Evaluates to False for {dtype} when passed as positional arg"


def test_validate_lf_with_transformation(test_lf):
    @validate_lf
    def test_func(lf):
        # Example transformation: filtering rows where 'a' > 1
        return lf.filter(pl.col("a") > 1)

    transformed_lf = test_func(lf=test_lf)
    assert isinstance(
        transformed_lf, pl.LazyFrame
    ), f"Expected LazyFrame, got {type(transformed_lf)}"
    # Collect the LazyFrame to a DataFrame and check the filter operation's result
    result_df = transformed_lf.collect()
    assert all(result_df["a"] > 1), f"Expected all 'a' > 1, got {result_df['a']}"
    assert len(result_df) == 2, f"Expected 1 row, got {len(result_df)}"
    assert result_df["a"].to_list() == [
        2,
        3,
    ], f"Expected [2, 3], got {result_df['a'].to_list()}"


def simple_decorator(func):
    """A simple test decorator that appends '_decorated' to the function's return value."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"{result}_decorated"

    return wrapper


@pytest.mark.parametrize("dtype", ["pd_df", "pl_df", "pl_lf", "np"])
def test_stacked_decorators(test_mapping, dtype):
    @simple_decorator
    @validate_lf
    def test_func(lf):
        # Here, we'll just return a simple string indicating success.
        # The outer decorator should append '_decorated' to this string.
        return "success"

    lf = test_mapping[dtype]
    result = test_func(lf=lf)
    assert (
        result == "success_decorated"
    ), "Decorators did not apply correctly or in the correct order"
