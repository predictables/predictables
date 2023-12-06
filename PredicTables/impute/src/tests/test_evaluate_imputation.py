import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from impute.src.evaluate_imputation import (
    evaluate_imputation_one_column,
    train_model_on_fold,
    calculate_fold_error,
    cross_validate_model,
    calculate_standard_error_of_mean,
    check_stopping_criterion,
)


@pytest.fixture
def data_setup():
    # Original DataFrame
    original_df = pd.DataFrame({"A": [1, 2, np.nan, 4, 5], "B": [5, np.nan, 7, 8, 9]})

    # Imputed DataFrame
    imputed_df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 6, 7, 8, 9]})

    # Missing mask
    missing_mask = original_df.isna()

    return imputed_df, original_df, missing_mask


@pytest.fixture
def mock_model():
    return LinearRegression()


############################# evaluate_imputation_one_column #############################


# Normal case
def test_evaluate_imputation_normal_case(data_setup):
    imputed_df, original_df, missing_mask = data_setup
    mae, mape = evaluate_imputation_one_column(
        imputed_df, original_df, missing_mask, "A"
    )
    assert mae == 0, f"MAE should be 0 for perfect imputation, not {mae}"
    assert (mape == 0) or (
        mape is None
    ), f"MAPE should be 0 when there are no non-zero original values, not {mape}"


# Column not in dataframe
def test_evaluate_imputation_column_not_found(data_setup):
    imputed_df, original_df, missing_mask = data_setup
    with pytest.raises(ValueError):
        evaluate_imputation_one_column(imputed_df, original_df, missing_mask, "C")


# No missing values
def test_evaluate_imputation_no_missing_values(data_setup):
    imputed_df, original_df, missing_mask = data_setup
    # Update original_df and imputed_df to have no missing values
    original_df.fillna(0, inplace=True)
    imputed_df.fillna(0, inplace=True)
    mae, mape = evaluate_imputation_one_column(
        imputed_df, original_df, missing_mask, "B"
    )
    assert mae == 0, f"MAE should be 0 when there are no missing values, not {mae}"
    assert mape == 0, f"MAPE should be 0 when there are no missing values, not {mape}"


# Incorrect data types
def test_evaluate_imputation_incorrect_data_types():
    with pytest.raises(AttributeError):
        evaluate_imputation_one_column(
            "not_a_dataframe", "not_a_dataframe", "not_a_dataframe", "A"
        )


############################# train_model_on_fold #############################


# Normal case
def test_train_model_on_fold_normal_case(mock_model, data_setup):
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    X_val = np.array([[5, 6]])

    predictions = train_model_on_fold(mock_model, X_train, y_train, X_val)

    assert (
        predictions is not None
    ), f"Predictions should not be None:\npredictions: {predictions}"
    assert (
        len(predictions) == len(X_val)
    ), f"Length of predictions should match validation set ({len(X_val)}), not {len(predictions)}"


# Empty training data
def test_train_model_on_fold_empty_train_data(mock_model):
    X_train = np.array([])
    y_train = np.array([])
    X_val = np.array([[5, 6]])

    with pytest.raises(ValueError):
        train_model_on_fold(mock_model, X_train, y_train, X_val)


# Incorrect data types
def test_train_model_on_fold_incorrect_data_types(mock_model):
    X_train = np.array(["not_an_array"]).reshape(-1, 1)
    y_train = np.array(["not_an_array"])
    X_val = np.array(["not_an_array"]).reshape(-1, 1)

    with pytest.raises(ValueError):
        train_model_on_fold(mock_model, X_train, y_train, X_val)


# Mismatched train & val sizes
def test_train_model_on_fold_mismatched_sizes(mock_model):
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2])
    X_val = np.array([[7, 8]])

    with pytest.raises(ValueError):
        train_model_on_fold(mock_model, X_train, y_train, X_val)


############################# calculate_fold_error #############################


# Normal case
def test_calculate_fold_error_normal_case():
    y_val = np.array([1, 2, 3])
    predictions = np.array([1, 2, 3])

    error = calculate_fold_error(y_val, predictions)
    assert error == 0, f"Error should be 0 for perfect predictions, not {error}"


# Mismatched sizes
def test_calculate_fold_error_mismatched_sizes():
    y_val = np.array([1, 2, 3])
    predictions = np.array([1, 2])  # Mismatched size

    with pytest.raises(ValueError):
        calculate_fold_error(y_val, predictions)


# Custom error metric
def test_calculate_fold_error_custom_metric():
    def absolute_error(y_true, y_pred):
        return np.abs(y_true - y_pred).mean()

    y_val = np.array([1, 2, 3])
    predictions = np.array([2, 3, 4])  # Error of 1 for each prediction

    error = calculate_fold_error(y_val, predictions, error_metric=absolute_error)
    assert (
        error == 1
    ), f"Error should be 1 for these predictions with absolute error metric, not {error}"


# Incorrect data types
def test_calculate_fold_error_incorrect_data_types():
    y_val = "not_an_array"
    predictions = "not_an_array"

    with pytest.raises(TypeError):
        calculate_fold_error(y_val, predictions)


############################# cross_validate_model #############################


# Normal case
def test_cross_validate_model_normal_case(mock_model):
    """This test checks if the function correctly performs cross-validation and returns a list of errors."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    n_folds = 2
    n_epochs = 5

    errors = cross_validate_model(mock_model, X, y, n_folds, n_epochs)
    assert isinstance(
        errors, list
    ), f"`errors` ({errors}) should return a list, not {type(errors)}"
    for i, e in enumerate(errors):
        assert isinstance(
            e, list
        ), f"List item {i}: {e}\nEach element in `errors` ({errors}) should be a list of errors for that epoch, not {type(e)}"
    assert (
        len(errors) == n_folds
    ), f"List of errors ({errors}) should contain errors for each fold (total of {n_folds} folds), but got {len(errors)}"
    for i, e in enumerate(errors):
        assert (
            len(e) == n_epochs
        ), f"List item {i}: {e}\nEach fold should contain errors for each epoch (total of {n_epochs} epochs), but got {len(e)} errors for this fold"

    # Check that errors are non-negative
    for i, e in enumerate(errors):
        for j, err in enumerate(e):
            assert (
                err >= 0
            ), f"List item {i}: {e}\nError {j}: {err}\nEach error should be non-negative, but got {err}"


# Invalid number of folds
def test_cross_validate_model_invalid_n_folds(mock_model):
    """This test ensures that the function handles invalid n_folds values correctly."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    n_folds = 0  # Invalid value
    n_epochs = 5

    with pytest.raises(ValueError):
        cross_validate_model(mock_model, X, y, n_folds, n_epochs)


# Custom error metric
def test_cross_validate_model_custom_error_metric(mock_model):
    """This test uses a custom error metric, similar to the one used in previous tests."""

    def absolute_error(y_true, y_pred):
        return np.abs(y_true - y_pred).mean()

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    n_folds = 2
    n_epochs = 5

    errors = cross_validate_model(
        mock_model, X, y, n_folds, n_epochs, error_metric=absolute_error
    )

    assert isinstance(errors, list), f"Should return a list, not {type(errors)}"
    assert (
        len(errors) == n_folds
    ), f"List should contain an error for each fold, but only contains {len(errors)} errors, with {n_folds} folds"


# Incorrect data types
def test_cross_validate_model_incorrect_data_types(mock_model):
    """This test checks the function's handling of incorrect data types."""
    X = "not_an_array"
    y = "not_an_array"
    n_folds = 2
    n_epochs = 5

    with pytest.raises(TypeError):
        cross_validate_model(mock_model, X, y, n_folds, n_epochs)


############################# calculate_standard_error_of_mean #############################


# Normal case
def test_calculate_standard_error_of_mean_normal_case():
    """This test checks the function with a standard list of errors."""
    errors = [[1, 2], [2, 3], [3, 4]]
    epoch = 1

    sem_result = calculate_standard_error_of_mean(errors, epoch)
    assert (
        sem_result >= 0
    ), f"Standard error of the mean should be a non-negative number, but got {sem_result}"


# Empty errors list
def test_calculate_standard_error_of_mean_empty_list():
    """This test ensures the function handles an empty errors list appropriately."""
    errors = []
    epoch = 0

    with pytest.raises(ValueError):
        calculate_standard_error_of_mean(errors, epoch)


# Incorrect data types
def test_calculate_standard_error_of_mean_incorrect_data_types():
    """This test checks the function's handling of incorrect data types."""
    errors = "not_a_list"
    epoch = 0

    with pytest.raises(TypeError):
        calculate_standard_error_of_mean(errors, epoch)


############################# check_stopping_criterion #############################


# Normal Case - Stopping Criterion Met
def test_check_stopping_criterion_met():
    """This test checks a scenario where the stopping criterion is met (errors in two consecutive epochs are within one standard error of each other)."""
    # Creating errors such that the last two are within one SEM of the third-last
    errors = [
        [1, 2, 1.5, 4.0, 4.1],  # Fold 1
        [1, 2, 3, 4.0, 4.2],  # Fold 2
        [1, 2, 5.5, 4.0, 4.1],  # Fold 3
    ]

    # Calculate SEM for the third-last epoch
    ave = (1.5 + 3 + 5.5) / 3
    var = ((1.5 - ave) ** 2 + (3 - ave) ** 2 + (5.5 - ave) ** 2) / 3
    std_err = var / np.sqrt(3)
    sem = calculate_standard_error_of_mean(errors, 2)
    assert np.isclose(
        sem, std_err
    ), f"SEM calculated by the function ({sem:.2f}) should match the manually calculated SEM ({std_err:.2f})"
    assert (
        sem == std_err
    ), f"SEM ({sem:.2f}) should be the square root of the variance ({np.sqrt(var):.2f}) divided by the square root of the number of samples ({np.sqrt(3):.2f}), or {std_err:.2f}, but got {sem:.2f}"

    upper_bound, lower_bound = ave + sem, ave - sem
    assert (
        4 <= upper_bound
    ), f"Average of 4th epoch ave(4, 4, 4) = 4 should be <= {upper_bound:.2f}"
    assert (
        np.mean(np.array([4.1, 4.2, 4.1])) <= upper_bound
    ), f"Average of 5th epoch ave(4.1, 4.2, 4.1) = {np.mean(np.array([4.1, 4.2, 4.1])):.2f} should be <= {upper_bound:.2f}"

    assert (
        4 >= lower_bound
    ), f"Average of 4th epoch ave(4, 4, 4) = 4 should be >= {lower_bound:.2f}"
    assert (
        np.mean(np.array([4.1, 4.2, 4.1])) >= lower_bound
    ), f"Average of 5th epoch ave(4.1, 4.2, 4.1) = {np.mean(np.array([4.1, 4.2, 4.1])):.2f} should be >= {lower_bound:.2f}"

    # Assuming the function is checking the last epoch
    assert check_stopping_criterion(
        errors
    ), "Stopping criterion should be met for converging errors"


# Normal Case - Stopping Criterion Not Met
def test_check_stopping_criterion_not_met():
    """This test verifies the scenario where the stopping criterion is not met."""
    errors = [[1, 2, 3, 3.1, 3.5], [1, 2, 3, 3.1, 4], [1, 2, 3, 3.1, 3.9]]
    assert not check_stopping_criterion(
        errors
    ), "Stopping criterion should not be met for diverging errors"


# Edge Case - Not Enough Data
def test_check_stopping_criterion_not_enough_data():
    """Test the function with fewer error values than needed for the check."""
    errors = [[1, 2]]
    assert not check_stopping_criterion(
        errors
    ), "Should return False when there is not enough data"


# Edge Case - Exact Boundary Values
def test_check_stopping_criterion_boundary_values():
    """Test with errors exactly at the boundary of the standard error."""
    errors = [[1, 2, 3, 3.1, 3.1], [1, 2, 3, 3.1, 3.1], [1, 2, 3, 3.1, 3.1]]
    assert check_stopping_criterion(
        errors
    ), "Stopping criterion should be met for errors exactly at the boundary"


def test_calculate_standard_error_of_mean_valid_input_NEW():
    # Test with valid input data
    errors = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    start_epoch = 1
    expected_sem = np.std([2, 3, 4]) / np.sqrt(3)
    assert calculate_standard_error_of_mean(errors, start_epoch) == expected_sem


def test_calculate_standard_error_of_mean_invalid_input_NEW():
    # Test with invalid input data
    with pytest.raises(ValueError):
        calculate_standard_error_of_mean([], 0)


def test_check_stopping_criterion_converged_NEW():
    # Test where errors have converged
    errors = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert check_stopping_criterion(errors) == True


def test_check_stopping_criterion_not_converged_NEW():
    # Test where errors have not converged
    errors = [[1, 2, 3], [1, 3, 4], [2, 4, 5]]
    assert check_stopping_criterion(errors) == False
