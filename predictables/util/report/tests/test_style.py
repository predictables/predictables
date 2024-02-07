import pytest
from predictables.util.report._Report import Report


@pytest.fixture
def report():
    # Setup fixture for Report instance
    return Report(filename="test_report.pdf")


@pytest.mark.parametrize(
    "tag, style",
    [
        ("h1", {"fontSize": 43}),
        ("h2", {"fontSize": 43}),
        ("h3", {"fontSize": 43}),
        ("h4", {"fontSize": 43}),
        ("h5", {"fontSize": 43}),
        ("h6", {"fontSize": 43}),
    ],
)
def test_style_updates_valid_attributes(report, tag, style):
    report = report.style(tag, **style)
    assert (
        getattr(report.styles.get(tag), "fontSize") == style["fontSize"]
    ), f"Expected {tag} to have fontSize {style['fontSize']}, got {getattr(report.styles.get(tag), 'fontSize')}"


def test_style_invalid_attributes(report):
    with pytest.warns(SyntaxWarning) as record:
        report = report.style("BodyText", fakeAttribute=123)

    assert "fakeAttribute is not a valid attribute of the stylesheet" in str(
        record.list[0].message
    ), "Expected warning for invalid attribute"


def test_style_returns_self(report):
    returned_report = report.style("BodyText", fontSize=12)
    assert (
        returned_report is report
    ), f"Expected style method to return self for chaining, but got {returned_report}"
