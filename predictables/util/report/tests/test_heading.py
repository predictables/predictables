import pytest
from predictables.util.report._Report import Report


@pytest.fixture
def report():
    # Setup fixture for Report instance
    return Report(filename="test_report.pdf")


@pytest.fixture
def heading_tag_generator(report):
    headings = enumerate(
        [
            report.h1("Heading 1"),
            report.h2("Heading 2"),
            report.h3("Heading 3"),
            report.h4("Heading 4"),
            report.h5("Heading 5"),
            report.h6("Heading 6"),
        ]
    )
    yield ((level, f"Heading {i + 1}", i + 1) for i, level in headings)


def test_heading_valid_attributes(report, heading_tag_generator):
    """Test setting valid attributes on Report instance."""
    for level, text, i in heading_tag_generator:
        assert level.text == text, f"Expected {level} to be {text}, got {level.text}"
        assert level.level == i, f"Expected {level} to be level {i}, got {level.level}"
        assert isinstance(
            level, Report
        ), f"Expected set method to return self, but got {level}"
