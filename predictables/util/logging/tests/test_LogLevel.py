import pytest
from predictables.util.logging._LogLevel import LogLevel

# Test data for from_str and _str methods
str_test_data = [
    ("I", "INFO"),
    ("INFO", "INFO"),
    ("D", "DEBUG"),
    ("DEBUG", "DEBUG"),
    ("W", "WARNING"),
    ("WARNING", "WARNING"),
    ("E", "ERROR"),
    ("ERROR", "ERROR"),
    ("C", "CRITICAL"),
    ("CRITICAL", "CRITICAL"),
    ("1", "INFO"),
    ("2", "DEBUG"),
    ("3", "WARNING"),
    ("4", "ERROR"),
    ("5", "CRITICAL"),
]

# Test data for from_int and _int methods
int_test_data = [
    (1, "INFO"),
    (2, "DEBUG"),
    (3, "WARNING"),
    (4, "ERROR"),
    (5, "CRITICAL"),
    (0, "INFO"),  # Test for values less than 1
    (6, "CRITICAL"),  # Test for values greater than 5
]

# Test data for get_str and str_ methods
get_str_test_data = [
    (LogLevel.INFO, "INFO"),
    (LogLevel.DEBUG, "DEBUG"),
    (LogLevel.WARNING, "WARNING"),
    (LogLevel.ERROR, "ERROR"),
    (LogLevel.CRITICAL, "CRITICAL"),
]

# Test data for get_int and int_ methods
get_int_test_data = [
    (LogLevel.INFO, 1),
    (LogLevel.DEBUG, 2),
    (LogLevel.WARNING, 3),
    (LogLevel.ERROR, 4),
    (LogLevel.CRITICAL, 5),
]


@pytest.mark.parametrize("input_str,expected_output", str_test_data)
def test_from_str(input_str, expected_output):
    assert LogLevel.from_str(input_str).name == expected_output


@pytest.mark.parametrize("input_str,expected_output", str_test_data)
def test__str(input_str, expected_output):
    assert LogLevel._str(input_str).name == expected_output  # noqa: SLF001


@pytest.mark.parametrize("input_int,expected_output", int_test_data)
def test_from_int(input_int, expected_output):
    assert LogLevel.from_int(input_int).name == expected_output


@pytest.mark.parametrize("input_int,expected_output", int_test_data)
def test__int(input_int, expected_output):
    assert LogLevel._int(input_int).name == expected_output  # noqa: SLF001


@pytest.mark.parametrize("log_level,expected_output", get_str_test_data)
def test_get_str(log_level, expected_output):
    assert log_level.get_str() == expected_output


@pytest.mark.parametrize("log_level,expected_output", get_str_test_data)
def test_str_(log_level, expected_output):
    assert log_level.str_() == expected_output


@pytest.mark.parametrize("log_level,expected_output", get_int_test_data)
def test_get_int(log_level, expected_output):
    assert log_level.get_int(log_level) == expected_output


@pytest.mark.parametrize("log_level,expected_output", get_int_test_data)
def test_int_(log_level, expected_output):
    assert log_level.int_(log_level) == expected_output