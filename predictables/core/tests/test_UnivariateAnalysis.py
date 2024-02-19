import pytest
import pandas as pd  # type: ignore
from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
from predictables.univariate import Univariate  # type: ignore


@pytest.fixture
def df_train():
    return pd.read_parquet("./cancer_train.parquet")


@pytest.fixture
def df_val():
    return pd.read_parquet("./cancer_val.parquet")


@pytest.fixture
def cv(df_train):
    return df_train["fold"]


@pytest.fixture
def test_data(df_train, df_val, cv):
    return df_train, df_val, cv


@pytest.fixture
def univariate(df_train, df_val):
    return Univariate(df_train, df_val, "fold", "worst_area", "y", False)


@pytest.fixture
def ua(df_train, df_val):
    return UnivariateAnalysis(
        "Cancer Model",
        df_train,
        df_val,
        "y",
        ["worst_area", "PC1"],
        False,
        "fold",
    )


def test_univariate_analysis_init(test_data):
    df_train, df_val, cv_folds = test_data
    model_name = "test_model"
    target_column_name = "y"
    feature_column_names = ["worst_area", "PC1"]
    time_series_validation = False
    cv_column_name = "fold"

    ua = UnivariateAnalysis(
        model_name=model_name,
        df_train=df_train,
        df_val=df_val,
        target_column_name=target_column_name,
        feature_column_names=feature_column_names,
        time_series_validation=time_series_validation,
        cv_column_name=cv_column_name,
        cv_folds=cv_folds,
    )

    # Verify basic attributes
    assert ua.model_name == model_name, f"Expected {model_name}, got {ua.model_name}"
    pd.testing.assert_frame_equal(ua.df, df_train)
    pd.testing.assert_frame_equal(ua.df_val, df_val)
    assert (
        ua.target_column_name == target_column_name
    ), f"Expected {target_column_name}, got {ua.target_column_name}"
    assert (
        ua.feature_column_names == feature_column_names
    ), f"Expected {feature_column_names}, got {ua.feature_column_names}"
    assert (
        ua.time_series_validation == time_series_validation
    ), f"Expected {time_series_validation}, got {ua.time_series_validation}"
    assert (
        ua.cv_column_name == cv_column_name
    ), f"Expected {cv_column_name}, got {ua.cv_column_name}"
    assert ua.cv_folds.equals(cv_folds), f"Expected {cv_folds}, got {ua.cv_folds}"


@pytest.mark.parametrize(
    "filename,default,expected",
    [
        (
            "Univariate Analysis Report.pdf",
            "Default Report",
            "Univariate Analysis Report",
        ),
        ("UnivariateAnalysisReport", "Default Report", "UnivariateAnalysisReport"),
        ("Different Report.pdf", "Univariate Analysis Report", "Different Report"),
        ("Complex.Report.Name.pdf", "Default Report", "Complex.Report.Name"),
        ("another_report_name", "Default Report", "another_report_name"),
    ],
)
def test_get_file_stem(ua, filename, default, expected):
    assert (
        ua._get_file_stem(filename, default) == expected
    ), f"Expected {expected} but got a different result"
