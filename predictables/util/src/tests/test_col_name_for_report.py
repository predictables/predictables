import pytest

from predictables.util.src._col_name_for_report import col_name_for_report


@pytest.mark.parametrize(
    "input_col_name, expected_output",
    [
        ("total_revenue_2020", "Total Revenue 2020"),
        ("cost_unit", "Cost Unit"),
        ("COLUMN_NAME", "Column Name"),
        (
            "column_name_with_multiple__underscores",
            "Column Name With Multiple Underscores",
        ),
        ("", ""),
        ("_", ""),
        ("log1p_total_revenue_2020", "log1p Total Revenue 2020"),
        ("log_1p_total_revenue_2020", "log 1p Total Revenue 2020"),
    ],
)
def test_col_name_for_report(input_col_name: str, expected_output: str):
    assert col_name_for_report(input_col_name) == expected_output, (
        f"Expected '{expected_output}' for input '{input_col_name}', but got "
        f"'{col_name_for_report(input_col_name)}'"
    )


@pytest.mark.parametrize(
    "invalid_value", [2, 100, 2.0, 100.0, 2.5, True, None, ["report"]]
)
def test_col_name_for_report_invalid(invalid_value: str):
    with pytest.raises(ValueError) as e:
        col_name_for_report(invalid_value)
    assert f"Invalid value {invalid_value} for column name." in str(e.value), (
        f"Expected 'Invalid value {invalid_value} for column name.' in the error message, "
        f"but got '{e.value}'"
    )
