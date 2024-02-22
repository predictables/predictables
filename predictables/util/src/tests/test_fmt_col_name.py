import pytest
from predictables.core.src._UnivariateAnalysis import fmt_col_name


@pytest.mark.parametrize(
    "input,output",
    [
        ("Total Revenue - 2020", "total_revenue_2020"),
        ("Net-Profit (After Tax)", "net_profit_after_tax"),
    ],
)
def test_fmt_col_name(input, output):
    result = fmt_col_name(input)

    assert result == output, f"Expected: {output} but got {result}"
    assert not result.endswith(
        "_"
    ), f"Expected: {output} (no trailing underscore), but got {result}"
    assert not result.startswith(
        "_"
    ), f"Expected: {output} (no leading underscore), but got {result}"
