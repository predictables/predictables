import pytest
from predictables.core.src._UnivariateAnalysis import _fmt_col_name


@pytest.mark.parametrize(
    "input,output",
    [
        ("Total Revenue - 2020", "total_revenue_2020"),
        ("Net-Profit (After Tax)", "net_profit_after_tax"),
    ],
)
def test_fmt_col_name(input, output):
    assert (
        _fmt_col_name(input) == output
    ), f"Expected: {output} but got {_fmt_col_name(input)}"
