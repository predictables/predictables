from reportlab.lib.pagesizes import inch, letter  # type: ignore
from reportlab.platypus import Paragraph  # type: ignore

from predictables.util.report._Report import Report


def test_report_shallow_copy():
    original_report = (
        Report(filename="test_report.pdf")
        .h1("Some Title")
        .h2("Some Subtitle")
        .h3("Some Sub-subtitle")
    )
    copied_report = original_report.copy()

    original_memory_addr = id(original_report)
    copied_memory_addr = id(copied_report)
    assert (
        original_memory_addr != copied_memory_addr
    ), f"Expected {original_memory_addr} to not be {copied_memory_addr}"

    # Check that the copied report is a new instance but not the same object
    assert (
        copied_report is not original_report
    ), f"Expected {copied_report} to not be {original_report}"
    assert (
        copied_report.filename == "test_report-COPY.pdf"
    ), f"Expected 'test_report-COPY.pdf', got {copied_report.filename}"
    assert (
        copied_report.elements is not original_report.elements
    ), f"Expected {copied_report.elements} to not be {original_report.elements}"
    assert (
        copied_report.elements[0] is original_report.elements[0]
    ), f"Expected {copied_report.elements[0]} to be {original_report.elements[0]}"
    assert (
        copied_report.elements[1] is original_report.elements[1]
    ), f"Expected {copied_report.elements[1]} to be {original_report.elements[1]}"

    # For shallow copies, changes to mutable elements in the copy affect the original
    copied_report.elements[0].text = "Changed title"
    assert (
        original_report.elements[0].text == "Changed title"
    ), f"Expected 'Changed title', got {original_report.elements[0].text}"

    # Ensure that the list itself is copied, but not the elements within it
    assert (
        copied_report.elements is not original_report.elements
    ), f"Expected {copied_report.elements} to not be {original_report.elements}"


def test_report_deep_copy():
    original_report = (
        Report(filename="test_report.pdf")
        .h1("Some Title")
        .h2("Some Subtitle")
        .h3("Some Sub-subtitle")
    )
    deep_copied_report = original_report.deepcopy()

    original_memory_addr = id(original_report)
    copied_memory_addr = id(deep_copied_report)
    assert (
        original_memory_addr != copied_memory_addr
    ), f"Expected {original_memory_addr} to not be {copied_memory_addr}"

    # Check that the deep copied report is a new instance
    assert (
        deep_copied_report is not original_report
    ), f"Expected {deep_copied_report} to not be {original_report}"
    assert (
        deep_copied_report.filename == "test_report-COPY.pdf"
    ), f"Expected 'test_report-COPY.pdf', got {deep_copied_report.filename}"

    # For deep copies, changes to mutable elements in the copy do not affect the original
    deep_copied_report.elements[
        0
    ].text = "this is a new element, distinct from the original"
    assert (
        original_report.elements[0].text != deep_copied_report.elements[0].text
    ), f"Expected {original_report.elements[0].text} to not be {deep_copied_report.elements[0].text}"

    # Ensure that both the list and the elements within it are copied
    assert (
        deep_copied_report.elements is not original_report.elements
    ), f"Expected {deep_copied_report.elements} to not be {original_report.elements}"
    assert (
        deep_copied_report.elements[0] is not original_report.elements[0]
    ), f"Expected {deep_copied_report.elements[0]} to not be {original_report.elements[0]}"
