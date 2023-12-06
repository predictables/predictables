import pytest
import numpy as np
import pandas as pd
from PredicTables.impute.IterativeRF_OLD import IterativeRF
from dataclasses import dataclass

@dataclass
class TestData:
    """Class to hold test data."""
    X: np.ndarray = None
    Xmask: np.ndarray  = None
    X_allna: np.ndarray = None
    X_nona: np.ndarray = None
    X_onecolna: np.ndarray = None
    X_onecol_allna: np.ndarray = None
    X_char: np.ndarray = None

    
    @classmethod
    def new(cls):
        cls.X = np.array([[1, 2, 3],
                           [4, np.nan, 6],
                           [7, 8, 9],
                           [np.nan, 11, 9],
                           [7, 8, np.nan]])
        cls.Xmask = np.array([[False, False, False],
                               [False, True, False],
                               [False, False, False],
                               [True, False, False],
                               [False, False, True]])
        cls.X_nona = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9],
                                [10, 11, 9],
                                [7, 8, 15]])
        cls.X_onecol_allna = np.array([[np.nan, 2, 3],
                                        [np.nan, 5, 6],
                                        [np.nan, 8, 9],
                                        [np.nan, 11, 9],
                                        [np.nan, 8, 15]])
        cls.X_onecolna = np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, np.nan, 9],
                                    [10, 11, 9],
                                    [7, 8, 15]])
        cls.X_allna = np.array([[np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan]])
        cls.X_char = np.array([['a', 'b', 'c'],
                                ['d', None, 'f'],
                                ['g', 'h', 'i'],
                                [None, 'k', 'i'],
                                ['g', 'k', None]])
        
        return cls
        
@pytest.fixture
def setup_data():
    """Fixture to provide sample data with missing values for testing."""
    return TestData.new()

@pytest.fixture
def setup_numerical_data():
    """Fixture to provide sample numerical data with missing values."""
    X = np.array([[1, 2, 3],
                  [4, np.nan, 6],
                  [7, 8, 9],
                  [np.nan, 11, 9],
                  [7, 8, np.nan]])
    return X

@pytest.fixture
def setup_categorical_data():
    """Fixture to provide sample categorical data with missing values."""
    X = np.array([['apple', 'banana', 'cherry'],
                  ['apple', None, 'banana'],
                  ['cherry', 'banana', 'apple'],
                  [None, 'cherry', 'banana'],
                  ['cherry', 'apple', None]], dtype=object)
    return X
    
def test_missing_mask(setup_data):
    """
    Test the _missing_mask method.
    
    The correct output should be a boolean matrix with 'True' where the original data had NaN values.
    """
    X = setup_data.X
    imputer = IterativeRF()
    mask = imputer._missing_mask(X)
    expected_mask = setup_data.Xmask
    assert np.array_equal(mask, expected_mask), \
        f"Missing mask not generated correctly: Actual mask:\n\n{mask}\
\n\n----------------------------------------------\n\nExpected mask:\n\n{expected_mask}"

    # Test mask for all NaN values
    X = setup_data.X_allna
    mask = imputer._missing_mask(X)
    expected_mask = np.ones(X.shape, dtype=bool)
    assert np.array_equal(mask, expected_mask), \
        f"Missing mask not generated correctly: Actual mask:\n\n{mask}\
\n\n----------------------------------------------\n\nExpected mask:\n\n{expected_mask}"

    # Test mask for no NaN values
    X = setup_data.X_nona
    mask = imputer._missing_mask(X)
    expected_mask = np.zeros(X.shape, dtype=bool)
    assert np.array_equal(mask, expected_mask), \
        f"Missing mask not generated correctly: Actual mask:\n\n{mask}\
\n\n----------------------------------------------\n\nExpected mask:\n\n{expected_mask}"

def test_median_impute(setup_data):
    """
    Test the _median_impute method for a specific column.
    
    The correct output should replace NaN values in the specified column with the median of that column.
    """
    medians = np.array([np.median(np.array([1,4,7,7])), np.median([2,8,11,8]), np.median(np.array([3,6,9,9]))])
    expected_X = np.array([[1, 2, 3],
                            [4, medians[1], 6],
                            [7, 8, 9],
                            [medians[0], 11, 9],
                            [7, 8, medians[2]]])

    rawX = setup_data.X
    imputer = IterativeRF()
    
    # Test each column individually
    for i in range(rawX.shape[1]):
        X = rawX.copy()
        X_imp = imputer._median_impute(X, i)
        assert np.array_equal(X_imp[:, i], expected_X[:, i]), \
            f"Median imputation not performed correctly: Actual column:\n\n{X_imp[:, i]}\
\n\n----------------------------------------------\n\nExpected column:\n\n{expected_X[:, i]}"
    
    # Test all columns at once
    X = rawX.copy()
    X_imp = imputer._median_impute(X)
    
    assert np.array_equal(X_imp, expected_X), \
        f"Median imputation not performed correctly: Actual matrix:\n\n{X_imp}\
\n\n----------------------------------------------\n\nExpected matrix:\n\n{expected_X}"


def test_mode_impute(setup_data):
    """
    Test the _mode_impute method for a specific column.
    
    The correct output should replace NaN values in the specified column with the mode of that column.
    """
    modes = [pd.Series([1,4,7,7]).mode()[0],
             pd.Series([2,8,11,8]).mode()[0],
             pd.Series([3,6,9,9]).mode()[0]]
    expected_X = np.array([[1, 2, 3],
                            [4, modes[1], 6],
                            [7, 8, 9],
                            [modes[0], 11, 9],
                            [7, 8, modes[2]]])
    imputer = IterativeRF()
    rawX = setup_data.X

    # Test each column individually
    for i in range(rawX.shape[1]):
        X = rawX.copy()
        X_imp = imputer._mode_impute(X, i)
        assert np.array_equal(X_imp[:, i], expected_X[:, i]), \
            f"Mode imputation not performed correctly: Actual column:\n\n{X_imp[:, i]}\
\n\n----------------------------------------------\n\nExpected column:\n\n{expected_X[:, i]}"
    
    # Test all columns at once
    X = rawX.copy()
    X_imp = imputer._mode_impute(X)
    assert np.array_equal(X_imp, expected_X), \
        f"Mode imputation not performed correctly: Actual matrix:\n\n{X_imp}\
\n\n----------------------------------------------\n\nExpected matrix:\n\n{expected_X}"
    
    # Test mode imputation for character data:
    # 1. each column individually:
    X = setup_data.X_char.copy()
    expected_X = np.array([['a', 'b', 'c'],
                            ['d', 'k', 'f'],
                            ['g', 'h', 'i'],
                            ['g', 'k', 'i'],
                            ['g', 'k', 'i']])
    for i in range(X.shape[1]):
        X_imp = imputer._mode_impute(X, i)
        assert np.array_equal(X_imp[:, i], expected_X[:, i]), \
            f"Mode imputation not performed correctly: Actual column:\n\n{X_imp[:, i]}\
\n\n----------------------------------------------\n\nExpected column:\n\n{expected_X[:, i]}"
        
    # 2. all columns at once:
    X = setup_data.X_char.copy()
    X_imp = imputer._mode_impute(X)
    assert np.array_equal(X_imp, expected_X), \
        f"Mode imputation not performed correctly: Actual matrix:\n\n{X_imp}\
\n\n----------------------------------------------\n\nExpected matrix:\n\n{expected_X}"
    



def test_determine_impute_order(setup_data):
    """
    Test the _determine_impute_order method.
    
    The correct output should return column indices sorted based on the number of missing values.
    """
    X = setup_data.X

    imputer = IterativeRF()

    # Standard data set has exactly one missing value per column, so
    # the order should be [0, 1, 2]
    order = imputer._determine_impute_order(X)
    expected_order = np.array([0, 1, 2])
    assert np.array_equal(order, expected_order), \
        f"Imputation order not determined correctly:\n\n\
Actual order: {order}\nExpected order: {expected_order}"
    
    # Mask two additional values in col 0 and 1 additional value in col 2
    X = setup_data.X.copy()
    X[0, 0] = np.nan
    X[1, 0] = np.nan
    X[2, 0] = np.nan

    # Now the order should be (in order of least missing values): 1, 2, 0
    order = imputer._determine_impute_order(X)
    expected_order = np.array([1, 2, 0])
    assert np.array_equal(order, expected_order), \
        f"Imputation order not determined correctly:\n\n\
Actual order: {order}\nExpected order: {expected_order}"
    
def test_initialize_impute_numerical(setup_numerical_data):
    """
    Test that `_initialize_impute` correctly imputes numerical columns with the median.
    """
    X = setup_numerical_data
    imputer = IterativeRF()
    col = 1  # Assuming column 1 is numerical with missing values
    X_imp = imputer._initialize_impute(X, col)
    # Calculate the expected median for the test
    median = np.median(X[~np.isnan(X[:, col]), col])
    # Confirm that NaNs are replaced with the median value in the specified column
    assert np.all(X_imp[~np.isnan(X[:, col]), col] == X[~np.isnan(X[:, col]), col]), \
        "Existing values have changed during median imputation."
    assert np.all(X_imp[np.isnan(X[:, col]), col] == median), \
        "NaN values have not been replaced with the median."

def test_initialize_impute_no_missing_values(setup_data):
    """
    Test that columns with no missing values are unchanged after imputation.
    """
    X = setup_data.X_nona.copy()  # Assuming this has no NaN values
    imputer = IterativeRF()
    col = 1  # Index of column to be tested
    original_column = X[:, col].copy()
    X_imp = imputer._initialize_impute(X, col)
    assert np.array_equal(X_imp[:, col], original_column), \
        "Column with no missing values should remain unchanged."

def test_initialize_impute_categorical(setup_categorical_data):
    """
    Test that `_initialize_impute` correctly imputes categorical columns with the mode.
    """
    X = setup_categorical_data
    imputer = IterativeRF()
    col = 1  # Assuming column 1 is categorical with missing values
    X_imp = imputer._initialize_impute(X, col)
    # Calculate the expected mode for the test
    mode = pd.Series(X[~pd.isnull(X[:, col]), col]).mode()[0]
    # Confirm that None are replaced with the mode value in the specified column
    assert np.all(X_imp[~pd.isnull(X[:, col]), col] == X[~pd.isnull(X[:, col]), col]), \
        "Existing values have changed during mode imputation."
    assert np.all(X_imp[pd.isnull(X[:, col]), col] == mode), \
        "None values have not been replaced with the mode."

def test_initialize_impute_empty_column_numerical():
    """
    Test that `_initialize_impute` handles an all NaN numerical column correctly.
    """
    X = np.array([[np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan]], dtype=float)
    imputer = IterativeRF()
    col = 1  # All values in this column are NaN
    X_imp = imputer._initialize_impute(X, col)
    # Should remain NaN because there is no median to calculate
    assert np.isnan(X_imp).all(), \
        f"All NaN numerical column was imputed incorrectly:\n\n{X_imp}"

# def test_initialize_impute_empty_column_categorical():
#     """
#     Test that `_initialize_impute` handles an all None categorical column correctly.
#     """
#     X = np.array([[None, None, None],
#                   [None, None, None],
#                   [None, None, None]], dtype=object)
#     imputer = IterativeRF()
#     col = 1  # All values in this column are None
#     X_imp = imputer._initialize_impute(X, col)
#     # Should remain None because there is no mode to calculate
#     assert pd.isnull(X_imp).all(), \
#         f"All None categorical column was imputed incorrectly:\n\n{X_imp}"
    
    
# def test_fit(setup_data):
#     """
#     Test the fit method.
    
#     After fitting, the imputer should have a model for each column, stored in the 'best_models' attribute.
#     """
#     X = setup_data.X
#     imputer = IterativeRF()
#     imputer.fit(X)
#     assert len(imputer.best_models) == X.shape[1], "Not all columns have been fitted with a model."

# def test_sequential_imputations(setup_data):
#     """
#     Test that multiple sequential imputations maintain data integrity.
#     """
#     X = setup_data.X.copy()
#     imputer = IterativeRF()
#     X_imp_first = imputer.fit_transform(X)
#     X_imp_second = imputer.transform(X)
#     # Check that the imputed values do not change after the first imputation
#     assert np.array_equal(X_imp_first, X_imp_second), \
#         f"Sequential imputations altered the imputed values:\n\n\
# First imputation: {X_imp_first}\nSecond imputation: {X_imp_second}"