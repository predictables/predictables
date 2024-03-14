import pytest
from reportlab.lib.pagesizes import letter  # type: ignore

from predictables.util.report._Report import Report


@pytest.fixture
def report():
    # Setup fixture for Report instance
    return Report(filename="test_report.pdf")


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("filename", "new_filename.pdf"),
        ("pagesize", letter),
        ("title", "New Title"),
        ("author", "New Author"),
        ("subject", "New Subject"),
        ("creator", "New Creator"),
        ("leftMargin", 100),
        ("rightMargin", 100),
        ("topMargin", 100),
        ("bottomMargin", 100),
    ],
)
def test_set_valid_attributes(report, attribute, value):
    """Test setting valid attributes on Report instance."""
    report = report.set(**{attribute: value})
    assert (
        getattr(report.doc, attribute) == value
    ), f"Expected {attribute} to be {value}, got {getattr(report.doc, attribute)}"
    assert isinstance(
        report, Report
    ), f"Expected set method to return self, but got {report}"


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("not_a_numeric_attribute", 123),
        ("not_a_string_attribute", "123"),
        ("not_a_boolean_attribute", True),
    ],
)
def test_set_invalid_attributes(report, attribute, value):
    """Test setting invalid attributes on Report instance."""
    with pytest.warns(SyntaxWarning):
        report.set(**{attribute: value})


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("filename", "new_filename.pdf"),
        ("pagesize", letter),
        ("title", "New Title"),
        ("author", "New Author"),
        ("subject", "New Subject"),
        ("creator", "New Creator"),
        ("leftMargin", 100),
        ("rightMargin", 100),
        ("topMargin", 100),
        ("bottomMargin", 100),
    ],
)
def test_set_returns_self(report, attribute, value):
    """Test that set method returns self."""
    assert (
        report.set(**{attribute: value}) is report
    ), f"Expected set method to return self when setting {attribute}, but got {report.set(**{attribute: value})}"
