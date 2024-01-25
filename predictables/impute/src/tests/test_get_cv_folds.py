import pandas as pd
import pytest

from predictables.impute.src._get_cv_folds import get_cv_folds


@pytest.fixture
def df():
    return pd.DataFrame({"A": range(10)})


def test_get_cv_folds_normal(df):
    folds = get_cv_folds(df, n_folds=5)

    assert len(folds) == len(df)  # Number of folds should match number of rows
    assert folds.nunique() == 5  # There should be 5 unique fold numbers


def test_get_cv_folds_invalid_n_folds(df):
    # Testing with invalid n_folds value
    with pytest.raises(ValueError):
        get_cv_folds(df, n_folds=0)  # n_folds must be greater than 0


# def test_get_cv_folds_return_indices():
#     # Testing the return_indices functionality
#     folds = get_cv_folds(df, n_folds=2, return_indices=True)
#     assert len(folds) == 2  # There should be 2 sets of indices for 2 folds
