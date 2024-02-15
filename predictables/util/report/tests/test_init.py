import pytest
from reportlab.lib.pagesizes import letter, inch  # type: ignore
from predictables.util.report._Report import Report


# Test initialization with default parameters
def test_report_init_default():
    report = Report(filename="test_report.pdf")
    assert (
        report.filename == "test_report.pdf"
    ), f"Expected 'test_report.pdf', got {report.filename}"
    assert report.pagesize == letter, f"Expected 'letter', got {report.pagesize}"
    assert report.dpi == 200, f"Expected 200, got {report.dpi}"
    assert (
        report.doc.leftMargin == 0.5 * inch
    ), f"Expected 0.5, got {report.doc.leftMargin}"
    assert (
        report.doc.rightMargin == 0.5 * inch
    ), f"Expected 0.5, got {report.doc.rightMargin}"
    assert (
        report.doc.topMargin == 0.5 * inch
    ), f"Expected 0.5, got {report.doc.topMargin}"
    assert (
        report.doc.bottomMargin == 0.5 * inch
    ), f"Expected 0.5, got {report.doc.bottomMargin}"


# Test initialization with custom margins
@pytest.mark.parametrize(
    "margins, expected",
    [
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([0.25, 0.5, 0.75, 1], [0.25, 0.5, 0.75, 1]),
    ],
)
def test_report_init_custom_margins(margins, expected):
    report = Report(filename="test_report.pdf", margins=margins)
    assert (
        report.doc.leftMargin == expected[0] * inch
    ), f"Expected {expected[0]}, got {report.doc.leftMargin}"
    assert (
        report.doc.rightMargin == expected[1] * inch
    ), f"Expected {expected[1]}, got {report.doc.rightMargin}"
    assert (
        report.doc.topMargin == expected[2] * inch
    ), f"Expected {expected[2]}, got {report.doc.topMargin}"
    assert (
        report.doc.bottomMargin == expected[3] * inch
    ), f"Expected {expected[3]}, got {report.doc.bottomMargin}"


# Test initialization with custom DPI
@pytest.mark.parametrize(
    "dpi, expected",
    [
        (100, 100),
        (300, 300),
    ],
)
def test_report_init_custom_dpi(dpi, expected):
    custom_dpi = dpi
    report = Report(filename="test_report.pdf", dpi=custom_dpi)
    assert report.dpi == expected, f"Expected {expected}, got {report.dpi}"


# Test initialization with custom pagesize
@pytest.mark.parametrize(
    "pagesize, expected",
    [
        ((8.5 * inch, 11 * inch), (8.5 * inch, 11 * inch)),
        ((7 * inch, 7 * inch), (7 * inch, 7 * inch)),
    ],
)
def test_report_init_custom_pagesize(pagesize, expected):
    custom_pagesize = pagesize
    report = Report(filename="test_report.pdf", pagesize=custom_pagesize)
    assert report.pagesize == expected, f"Expected {expected}, got {report.pagesize}"


# Assuming additional functionality for error handling is added, test incorrect margins length
def test_report_init_incorrect_margins_length():
    with pytest.raises(IndexError):
        Report(filename="test_report.pdf", margins=[0.5, 0.5])
