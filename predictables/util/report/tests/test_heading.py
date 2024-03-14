import pytest

from predictables.util.report._Report import Report


@pytest.fixture
def report():
    # Setup fixture for Report instance
    return Report(filename="test_report.pdf")


@pytest.mark.parametrize(
    "heading,style,text,level",
    [
        ("h1", {"fontSize": 43}, "Heading 1", 1),
        ("h2", {"fontSize": 43}, "Heading 2", 2),
        ("h3", {"fontSize": 43}, "Heading 3", 3),
        ("h4", {"fontSize": 43}, "Heading 4", 4),
        ("h5", {"fontSize": 43}, "Heading 5", 5),
        ("h6", {"fontSize": 43}, "Heading 6", 6),
    ],
)
def test_heading_valid_attributes(report, heading, style, text, level):
    """Test setting valid heading 1 attributes on Report instance."""
    rpt1 = report.heading(level, text).style(heading, **style)
    rpt2 = report
    setattr(rpt2, heading, text)
    rpt2 = rpt2.style(heading, **style)

    assert (
        rpt1.styles.get(heading).fontSize == style["fontSize"]
    ), f"Expected {heading} to have fontSize {style['fontSize']}, got {rpt1.styles.get(heading).fontSize}"
    assert (
        rpt2.styles.get(heading).fontSize == style["fontSize"]
    ), f"Expected {heading} to have fontSize {style['fontSize']}, got {rpt2.styles.get(heading).fontSize}"
    assert isinstance(
        rpt1, Report
    ), f"Expected set method to return self (or at least a Report object), but got {rpt1}, a {type(rpt1)} object"
    assert isinstance(
        rpt2, Report
    ), f"Expected set method to return self (or at least a Report object), but got {rpt2}, a {type(rpt2)} object"
    assert len(rpt1.elements) == 1, f"Expected 1 element, got {len(rpt1.elements)}"
    assert len(rpt2.elements) == 1, f"Expected 1 element, got {len(rpt2.elements)}"
    assert (
        rpt1.elements[0].text == text
    ), f"Expected {text} for rpt1, got {rpt1.elements[0].text}"
    assert (
        rpt2.elements[0].text == text
    ), f"Expected {text} for rpt2, got {rpt2.elements[0].text}"
