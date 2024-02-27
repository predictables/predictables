import polars as pl

from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    _rolling_sum_no_categories,
)


def test_rolling_sum_no_categories():
    # Sample data similar to what you've provided
    data = {
        "date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-03-01"],
        "value": [10, 15, 20, 30],
    }
    lf = pl.DataFrame(data).lazy().with_columns(pl.col("date").str.strptime(pl.Date))

    # Apply the rolling sum function
    pl.Config.set_verbose(True)
    with pl.Config(verbose=True):
        result_lf = _rolling_sum_no_categories(
            lf=lf,
            date_col="date",
            x="value",
            x_name="aggregated_value",
            every="1d",
            period="1y",
            offset="-1mo",
        )

        # Attempt to collect the result
        result_df = result_lf.collect()

    # Assertions to validate the results
    # These should be adapted based on the expected outcome of your function
    assert "date" in result_df.columns, "Date column should be present in the result."
    assert (
        "aggregated_value" in result_df.columns
    ), "Aggregated value column should be present in the result."
    # Add more assertions as needed based on the expected behavior of your function


# Run the test with pytest from the command line
# pytest test_rolling_sum.py
