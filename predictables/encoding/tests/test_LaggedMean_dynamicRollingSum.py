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
    _formatted_category_cols,
    _reversed_date_col,
    _group_by_no_categories,
)


@pytest.fixture
def csv():
    date_fmt_str = "%m/%d/%Y"
    return pl.scan_csv(
        "predictables/encoding/tests/rolling_date_example.csv"
    ).with_columns(
        [
            pl.col("date").str.to_date(format=date_fmt_str).cast(pl.Date).name.keep(),
            pl.col("incr_value").cast(pl.Float64).name.keep(),
            pl.col("month_prior")
            .str.to_date(format=date_fmt_str)
            .cast(pl.Date)
            .name.keep(),
            pl.col("year_prior")
            .str.to_date(format=date_fmt_str)
            .cast(pl.Date)
            .name.keep(),
            pl.col("rolling_sum").cast(pl.Float64).name.keep(),
            pl.col("date")
            .str.to_date(format=date_fmt_str)
            .cast(pl.Date)
            .alias("cur_date"),
            pl.col("cat1").cast(pl.Utf8).cast(pl.Categorical).name.keep(),
            pl.col("cat2").cast(pl.Utf8).cast(pl.Categorical).name.keep(),
        ]
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
        .with_columns(
            [
                pl.Series("category", "ababaabbab".split())
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .alias("category"),
                pl.Series("category1", "ccddccddcc".split())
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .alias("category1"),
                pl.Series("category2", "eeffgghhii".split())
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .alias("category2"),
            ]
        )
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


def test_get_date_map_on_csv(csv):
    result = _get_date_map(csv, "date")
    min_date = csv.collect().select(pl.col("date").min()).to_series()[0]
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


def test_handle_cat_input_single_string():
    category_cols = "category"
    expected_output = ["category"]
    assert (
        _handle_cat_input(category_cols) == expected_output
    ), f"Expected {expected_output} does not match the result {_handle_cat_input(category_cols)}."


def test_handle_cat_input_list_of_strings():
    category_cols = ["category1", "category2"]
    expected_output = ["category1", "category2"]
    assert (
        _handle_cat_input(category_cols) == expected_output
    ), f"Expected {expected_output} does not match the result {_handle_cat_input(category_cols)}."


def test_handle_cat_input_none():
    category_cols = None
    expected_output = []
    assert (
        _handle_cat_input(category_cols) == expected_output
    ), f"Expected {expected_output} does not match the result {_handle_cat_input(category_cols)}."


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
            ["category1", "category2"],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ),  # Test with multiple category_cols
    ],
)
def test_get_original_order(
    create_test_frame, date_col, category_cols, expected_output
):
    result = _get_original_order(create_test_frame, date_col, category_cols)

    # Test the order itself
    assert (
        result.collect().to_series().to_list() == expected_output
    ), f"Expected order {expected_output} does not match the calculated order {result.collect().to_series().to_list()}."


@pytest.mark.parametrize(
    "date_col, category_cols, expected_output",
    [
        (
            "date",
            None,
            [1825 - i for i in range(1826)],
        ),
        (
            "date",
            ["cat1"],
            [1825 - i for i in range(1826)],
        ),
        (
            "date",
            ["cat1", "cat2"],
            [1825 - i for i in range(1826)],
        ),
    ],
)
def test_get_original_order_with_csv(csv, date_col, category_cols, expected_output):
    result = _get_original_order(csv, date_col, category_cols)

    # Test the order itself
    assert (
        result.collect().to_series().to_list() == expected_output
    ), f"Expected order {expected_output} does not match the calculated order {result.collect().to_series().to_list()}."


@pytest.mark.parametrize(
    "category_cols",
    [
        123,  # Test with integer
        12.34,  # Test with float
        True,  # Test with boolean
    ],
)
def test_formatted_category_cols_type_errors(category_cols):
    with pytest.raises(TypeError):
        _formatted_category_cols(category_cols)


# def test_groupby_no_categories_csv(csv):
#     reversed_col = _reversed_date_col(csv, "date")
#     df = _group_by_no_categories(csv, reversed_col, "incr_value", "result").all()
#     expected = csv.select("rolling_sum").collect()["rolling_sum"]
#     result = df.collect()["result"]
#     index = df.with_row_index().collect()["row_index"]
#     for i, val in enumerate(result):
#         assert (
#             val == expected[i]
#         ), f"Expected {expected[i]} does not match the result {val} at index {index[i]}."


# def test_dynamic_rolling_sum_no_categories(csv):
#     lf = csv.collect()

#     # Convert date column for the lazyframe processing
#     lf = lf.with_columns([pl.col("date").cast(pl.Date).name.keep()])

#     # Expected rolling_sum from the CSV
#     expected_rolling_sum = lf.select("rolling_sum").to_series()

#     # Calculating the rolling sum using the dynamic_rolling_sum function
#     calculated_rolling_sum = dynamic_rolling_sum(
#         lf=lf.lazy(),
#         x="incr_value",
#         date_col="date",
#         x_name="calculated_rolling_sum",
#         every="1d",
#         period="1y",
#         offset="-1mo",
#     )

#     assert all(
#         calculated_rolling_sum == expected_rolling_sum
#     ), "Calculated rolling sums do not match expected values."


def test_group_by_no_categories():
    # Create a sample LazyFrame with dates and values
    data = {"date": ["2023-01-01", "2023-02-01", "2023-03-01"], "value": [10, 20, 30]}
    lf = pl.DataFrame(data).lazy()
    lf = lf.with_columns([pl.col("date").str.strptime(pl.Date).name.keep()])

    # Simulate reversed_date_expr for testing purposes
    reversed_date_expr = _reversed_date_col(lf, "date")

    # Apply the _group_by_no_categories function with mocked parameters
    grouped_lf = _group_by_no_categories(
        lf=lf,
        reversed_date_expr=reversed_date_expr,
        x="value",
        x_name="grouped_value",
        every="1mo",
        period="2mo",
        offset="0mo",
    )

    assert grouped_lf is not None, "Expected a non-None result from grouping operation"


def test_group_by_no_categories_with_aggregations():
    # Create a sample LazyFrame with dates and values
    data = {
        "date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-03-01"],
        "value": [10, 15, 20, 30],
    }
    lf = pl.DataFrame(data).lazy()
    lf = lf.with_columns(pl.col("date").str.strptime(pl.Date))

    # Define reversed_date_expr as the date column itself for simplicity
    reversed_date_expr = _reversed_date_col(lf, "date")

    # Define a wrapper around _group_by_no_categories to include aggregations
    def apply_group_with_aggregations(
        lf, reversed_date_expr, x, x_name, every, period, offset
    ):
        # Group using the provided function
        grouped_lf = _group_by_no_categories(
            lf=lf,
            reversed_date_expr=reversed_date_expr,
            x=x,
            x_name=x_name,
            every=every,
            period=period,
            offset=offset,
        )

        # Apply aggregations
        return grouped_lf.agg(
            [
                pl.col("date"),
                pl.sum("value").alias("sum"),
                pl.min("value").alias("min"),
                pl.max("value").alias("max"),
                pl.count("value").alias("count"),
                pl.mean("value").alias("mean"),
            ]
        )

    # Apply the grouping and aggregation
    result_lf = apply_group_with_aggregations(
        lf=lf,
        reversed_date_expr=reversed_date_expr,
        x="value",
        x_name="aggregated_value",
        every="1mo",
        period="2mo",
        offset="0mo",
    ).collect()

    # Expected results based on the data and aggregation definitions
    expected_data = [
        {
            "date": "2023-01-01",
            "sum": 25,
            "min": 10,
            "max": 15,
            "count": 2,
            "mean": 12.5,
        },
        {
            "date": "2023-02-01",
            "sum": 20,
            "min": 20,
            "max": 20,
            "count": 1,
            "mean": 20.0,
        },
        {
            "date": "2023-03-01",
            "sum": 30,
            "min": 30,
            "max": 30,
            "count": 1,
            "mean": 30.0,
        },
    ]
    expected_df = pl.DataFrame(expected_data)

    # Compare the result to the expected DataFrame
    assert result_lf.frame_equal(
        expected_df, null_equal=True
    ), "Aggregated results do not match expected values."
