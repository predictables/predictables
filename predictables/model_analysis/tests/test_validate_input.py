import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from predictables.model_analysis import validate_input

# Mocking a simple dataset and models for testing
X_valid = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
X_invalid = [1, 2, 3]  # Not a DataFrame

y_valid = pd.Series([1, 0, 1])
y_invalid = [1, 0, 1]  # Not a Series

model_valid = RandomForestClassifier()
model_invalid = "not_a_model"  # Does not have fit and predict methods


# Defining test functions
def test_validate_input_with_valid_inputs():
    result, message = validate_input(X_valid, y_valid, model_valid)
    assert result, "Test should have passed for valid inputs (Test 1)."


def test_validate_input_with_invalid_dataset():
    result, message = validate_input(X_invalid, y_valid, model_valid)
    assert (
        not result
    ), f"Test should have failed for invalid dataset {X_invalid} (Test 2)."


def test_validate_input_with_invalid_model():
    result, message = validate_input(X_valid, y_valid, model_invalid)
    assert not result, f"Should have failed for invalid model {model_invalid} (Test 3)."


def test_validate_input_with_invalid_target():
    result, message = validate_input(X_valid, y_invalid, model_valid)
    assert not result, f"Should have failed for invalid target {y_invalid} (Test 4)."
