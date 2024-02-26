# test_dynamic_rolling_sum.py
import datetime
import polars as pl
import pytest

from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    _get_date_map,
    _handle_cat_input,
    _get_x_name,
    dynamic_rolling_sum,
    _get_original_order,
)


@pytest.fixture
def create_test_frame():
    """Helper function to create a basic test LazyFrame with a date column."""
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2020, 10, 1)
    date_range = (
        pl.datetime_range(start=start_date, end=end_date, interval="1mo")
        .cast(pl.Date)
        .alias("date")
    )
    return (
        pl.DataFrame({"x": [10 * (i + 1) for i in range(10)]})
        .lazy()
        .with_columns(date_range)
    )


@pytest.fixture
def rolling_sum_test_frame():
    """
    Creates a test LazyFrame with dates and corresponding values for testing the
    dynamic_rolling_sum function.
    """
    # Define the start date for the data
    start_date = datetime.datetime(2020, 1, 1)

    # Generate a date range and corresponding values
    dates = [
        start_date + datetime.timedelta(days=i) for i in range(10)
    ]  # 10 days of data
    values = [i for i in range(10)]  # Simple incremental values

    # Create and return the LazyFrame
    df = pl.DataFrame({"date": dates, "x": values})

    return df.lazy()


# @pytest.mark.parametrize("x_name", [None, "custom_name"])
# @pytest.mark.parametrize("expected_output", [[0, 1, 3, 6, 10, 15, 21, 28, 36, 45]])
# def test_dynamic_rolling_sum_no_categories(
#     rolling_sum_test_frame, x_name, expected_output
# ):
#     """
#     Test the dynamic_rolling_sum function without category columns.
#     """
#     # Prepare the parameters
#     date_col = "date"
#     x = "x"
#     category_cols = None  # Explicitly stating no categories for clarity

#     # Execute the function under test
#     result_series = dynamic_rolling_sum(
#         lf=rolling_sum_test_frame,
#         x=x,
#         date_col=date_col,
#         category_cols=category_cols,
#         x_name=x_name,
#     )

#     # Verify the results
#     for i in range(len(result_series)):
#         assert (
#             result_series.to_list()[i] == expected_output[i]
#         ), f"Expected rolling sum {expected_output[i]} does not match the calculated rolling sum {result_series.to_list()[i]}."


def test_get_date_map(create_test_frame):
    result = _get_date_map(create_test_frame, "date")
    min_date = create_test_frame.collect().select(pl.col("date").min()).to_series()[0]
    max_date = datetime.date.today()

    # Generate the full date range from min_date to today, similar to _get_date_map logic
    full_date_range = [
        min_date + datetime.timedelta(days=x)
        for x in range((max_date - min_date).days + 1)
    ]
    reversed_date_range = list(reversed(full_date_range))

    # Create the expected mapping based on the logic of reversing the date range
    expected_map = {
        original: reversed
        for original, reversed in zip(full_date_range, reversed_date_range)
    }

    # Verify each date in the generated map matches the expected reversed date map
    for date, reversed_date in expected_map.items():
        assert date in result, f"Date {date} is not in the result map."
        assert (
            result[date] == reversed_date
        ), f"Date {date} is not mapped to its correct reversed counterpart {reversed_date}."


# @pytest.mark.parametrize(
#     "expected_sum",
#     [
#         # Expected sums for each row considering a rolling year
#         [0, 10, 30, 60, 100, 150],
#     ],
# )
# def test_basic_rolling_sum(create_test_frame, expected_sum):
#     result_series = dynamic_rolling_sum(create_test_frame, "x", "date")
#     assert (
#         result_series.to_list() == expected_sum
#     ), "The calculated rolling sums do not match the expected values."


def test_handle_cat_input_single_string():
    category_cols = "category"
    expected_output = ["category"]
    assert _handle_cat_input(category_cols) == expected_output


def test_handle_cat_input_list_of_strings():
    category_cols = ["category1", "category2"]
    expected_output = ["category1", "category2"]
    assert _handle_cat_input(category_cols) == expected_output


def test_handle_cat_input_none():
    category_cols = None
    expected_output = []
    assert _handle_cat_input(category_cols) == expected_output


def test_handle_cat_input_not_string_or_list():
    category_cols = 123
    with pytest.raises(TypeError):
        _handle_cat_input(category_cols)


@pytest.mark.parametrize(
    "x, x_name, expected_output",
    [
        ("sales", None, "sales_rolling_sum"),  # Test with x_name as None
        ("sales", "monthly_sales_sum", "monthly_sales_sum"),  # Test with valid x_name
    ],
)
def test_get_x_name(x, x_name, expected_output):
    assert _get_x_name(x, x_name) == expected_output


@pytest.mark.parametrize(
    "x", [None, 123, 123.45, True, False, [1, 2, 3], {"a": 1, "b": 2}]
)
def test_invalid_x_name_type(x):
    with pytest.raises(TypeError):
        _get_x_name(x, "not good enough to make a name")


@pytest.mark.parametrize("x", ["", " ", "  ", "   ", "    ", "\n", "\t", "\r", "\r\n"])
@pytest.mark.parametrize("x_name", ["a_name", None, ""])
def test_empty_x_name(x, x_name):
    with pytest.raises(ValueError):
        _get_x_name(x, x_name)


@pytest.mark.parametrize(
    "x_name", [123, 123.45, True, False, [1, 2, 3], {"a": 1, "b": 2}]
)
def test_xname_typeerror(x_name):
    with pytest.raises(TypeError):
        _get_x_name("sales", x_name)


@pytest.mark.parametrize(
    "date_col, category_cols, expected_output",
    [
        (
            "date",
            None,
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ),  # Test with category_cols as None
        (
            "date",
            ["category"],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ),  # Test with valid category_cols
        (
            "date",
            ["category", "category2"],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ),  # Test with multiple category_cols
    ],
)
def test_get_original_order(
    create_test_frame, date_col, category_cols, expected_output
):
    result = _get_original_order(create_test_frame, date_col, category_cols).select(
        [pl.col("index")]
    )
    assert result.collect()["index"].to_list() == expected_output
