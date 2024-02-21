from predictables.univariate.src.plots.util._plot_label import plot_label


def test_plot_label_with_spaces():
    # Test when the input string has spaces
    input_str = "hello world"
    expected_output = "[Hello World]"
    assert (
        plot_label(input_str) == expected_output
    ), f"Plot label not formatted correctly: expected: {expected_output}, got: {plot_label(input_str)}"


def test_plot_label_with_underscores():
    # Test when the input string has underscores
    input_str = "hello_world"
    expected_output = "[Hello World]"
    assert (
        plot_label(input_str) == expected_output
    ), f"Plot label not formatted correctly: expected: {expected_output}, got: {plot_label(input_str)}"


def test_plot_label_with_brackets():
    # Test when the input string starts with a bracket
    input_str = "[hello_world]"
    expected_output = "[Hello World]"
    assert (
        plot_label(input_str) == expected_output
    ), f"Plot label not formatted correctly: expected: {expected_output}, got: {plot_label(input_str)}"


def test_plot_label_with_empty_string():
    # Test when the input string is empty
    input_str = ""
    expected_output = ""
    assert (
        plot_label(input_str) == expected_output
    ), f"Plot label not formatted correctly: expected: {expected_output}, got: {plot_label(input_str)}"


def test_plot_label_with_single_character():
    # Test when the input string has a single character
    input_str = "a"
    expected_output = "[A]"
    assert (
        plot_label(input_str) == expected_output
    ), f"Plot label not formatted correctly: expected: {expected_output}, got: {plot_label(input_str)}"


def test_plot_label_without_brackets():
    # Test when incl_bracket is set to False
    input_str = "hello_world"
    expected_output = "Hello World"
    assert plot_label(input_str, incl_bracket=False) == expected_output, (
        f"Plot label not formatted correctly: expected: {expected_output}, "
        f"got: {plot_label(input_str, incl_bracket=False)}"
    )
