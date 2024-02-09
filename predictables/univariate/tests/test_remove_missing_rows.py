import polars as pl
from predictables.univariate.src._remove_missing_rows import remove_missing_rows


def test_remove_missing_rows():
    # Create a test dataframe
    df = pl.DataFrame(
        {"feature": [1, 2, None, 4, 5], "target": [None, 7, 8, 9, None]}
    ).lazy()

    # Call the function
    result = remove_missing_rows(df, "feature", "target")

    # Check the result
    expected = pl.DataFrame({"feature": [2, 4], "target": [7, 9]}).lazy()

    (
        pl.testing.assert_frame_equal(result, expected),
        f"Expected:\n{expected}\n\nGot:\n{result}",
    )


def test_remove_missing_rows_no_missing_values():
    # Create a test dataframe
    df = pl.DataFrame({"feature": [1, 2, 3, 4, 5], "target": [6, 7, 8, 9, 10]}).lazy()

    # Call the function
    result = remove_missing_rows(df, "feature", "target")

    # Check the result (should be the same as the input dataframe)
    pl.testing.assert_frame_equal(result, df), f"Expected:\n{df}\n\nGot:\n{result}"


def test_remove_missing_rows_all_missing_values():
    # Create a test dataframe
    df = pl.DataFrame(
        {
            "feature": [None, None, None, None, None],
            "target": [None, None, None, None, None],
        }
    ).lazy()

    # Call the function
    result = remove_missing_rows(df, "feature", "target")

    # Assert that the resulting DataFrame is empty
    assert (
        len(result.collect()) == 0
    ), f"Expected an empty DataFrame, but got:\n{result.collect()}"