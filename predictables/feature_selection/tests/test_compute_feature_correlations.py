import polars as pl
import polars.testing as pltest
import pandas as pd
import pytest


from predictables.feature_selection.src.backward_stepwise import (
    compute_feature_correlations,
)


@pytest.mark.parametrize(
    "input_data,expected_output",
    [
        # Test with normal numeric data
        (
            pl.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1], "C": [1, 2, 2]}).lazy(),
            [
                ("A", "B", -1.0),
                ("A", "C", 0.8660254037844387),
                ("B", "C", -0.8660254037844387),
            ],
        ),
        # Test with non-numeric data (should skip non-numeric columns)
        (
            pl.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1, 2, 2]}).lazy(),
            [("A", "C", 0.8660254037844387)],
        ),
        # # Test with an empty dataframe
        # (pl.DataFrame({}).lazy(), []),
        # # Test with a single column
        # (pl.DataFrame({"A": [1, 2, 3]}).lazy(), []),
    ],
)
def test_compute_feature_correlations(input_data, expected_output):
    # Convert expected output to DataFrame and sort by correlation for comparison
    expected_df = (
        pl.from_pandas(
            pd.DataFrame(expected_output, columns=["col1", "col2", "correlation"])
        )
        .lazy()
        .sort("correlation", descending=True)
    )
    result_df = compute_feature_correlations(input_data)
    pltest.assert_frame_equal(result_df, expected_df)
