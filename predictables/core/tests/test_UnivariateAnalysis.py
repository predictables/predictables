import pandas as pd  # type: ignore
import pytest

from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
from predictables.univariate import Univariate
from predictables.util import to_pd_df, to_pd_s


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
        "Cancer Model", df_train, df_val, "y", ["worst_area", "pc1"], False, "fold"
    )


def test_univariate_analysis_init(test_data):
    df_train, df_val, cv_folds = test_data
    model_name = "test_model"
    target_column_name = "y"
    feature_column_names = ["worst_area", "pc1"]
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
    print(ua.df.columns)
    pd.testing.assert_frame_equal(
        to_pd_df(ua.df)
        .drop(columns=["log1p_worst_area", "log1p_pc1"])
        .reset_index(drop=True),
        to_pd_df(df_train).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        to_pd_df(ua.df_val)
        .drop(columns=["log1p_worst_area", "log1p_pc1"])
        .reset_index(drop=True),
        to_pd_df(df_val).reset_index(drop=True),
    )
    assert (
        ua.target_column_name == target_column_name
    ), f"Expected {target_column_name}, got {ua.target_column_name}"
    assert (
        ua.feature_column_names[:-2] == feature_column_names
    ), f"Expected {feature_column_names}, got {ua.feature_column_names[:-2]}"
    assert (
        ua.time_series_validation == time_series_validation
    ), f"Expected {time_series_validation}, got {ua.time_series_validation}"
    assert (
        ua.cv_column_name == cv_column_name
    ), f"Expected {cv_column_name}, got {ua.cv_column_name}"
    assert to_pd_s(ua.cv_folds).equals(
        to_pd_s(cv_folds)
    ), f"Expected {to_pd_s(cv_folds)}, got {to_pd_s(ua.cv_folds)}"


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


@pytest.mark.parametrize(
    "file_stem,file_num_start_num,end_num,default,expected",
    [
        ("analysis", None, None, "Univariate Analysis Report", "analysis.pdf"),
        ("report", 1, 2, "Univariate Analysis Report", "report_1_3.pdf"),
        (None, None, None, "Univariate Analysis Report", "Univariate Analysis Report"),
        ("only_stem", 1, None, "Univariate Analysis Report", "only_stem.pdf"),
        ("only_stem", None, 2, "Univariate Analysis Report", "only_stem.pdf"),
    ],
)
def test_rpt_filename(ua, file_stem, file_num_start_num, end_num, default, expected):
    result = ua._rpt_filename(
        file_stem=file_stem,
        start_num=file_num_start_num,
        end_num=end_num,
        default=default,
    )
    assert result == expected, f"Expected filename '{expected}', but got '{result}'"


@pytest.mark.parametrize(
    "total_features,max_per_file,expected",
    [
        (
            100,
            10,
            "Building 100 univariate analysis reports,and packaging in increments of 10",
        ),
        (
            50,
            25,
            "Building 50 univariate analysis reports,and packaging in increments of 25",
        ),
        (
            1,
            1,
            "Building 1 univariate analysis reports,and packaging in increments of 1",
        ),
        (
            0,
            10,
            "Building 0 univariate analysis reports,and packaging in increments of 10",
        ),
    ],
)
def test_build_desc(total_features, max_per_file, expected):
    description = UnivariateAnalysis._build_desc(total_features, max_per_file)
    assert (
        description == expected
    ), f"Expected description '{expected}', but got '{description}'"
