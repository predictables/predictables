import pytest
import os
from predictables.fit_catboost_time_series import get_file_list


@pytest.fixture
def fake_fs():
    """Create a fake filesystem to use to test the function.

    It should have "root"-level files, and subfolder/subdirectory files.
    """
    return ["file1", "file2", "folder1", "folder2"]


@pytest.mark.parametrize(
    "expected_result",
    [
        pytest.param([], id="Empty list when no files"),
        pytest.param(["file1", "file2"], id="List of files when files exist"),
    ],
)
def test_get_file_list(mocker, expected_result):
    mocker.patch("os.listdir", return_value=expected_result)
    assert get_file_list() == expected_result