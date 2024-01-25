import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from predictables.univariate.src._time_series_validation_filter import (
    time_series_validation_filter,
)


# Define a synthetic dataset
@pytest.fixture
def df():
    # Create a synthetic dataset using pandas
    return pd.DataFrame(
        {
            "feature_col": list(range(50)),
            "target_col": [i * 10 for i in range(50)],
            "cv": 15 * [1] + 15 * [2] + 15 * [3] + 5 * [-1],
        }
    )


@pytest.fixture
def df_val(df):
    return df.loc[df.cv.eq(-1)].reset_index(drop=True)


@pytest.fixture
def df_train(df):
    return df.loc[df.cv.ne(-1)].reset_index(drop=True)


@pytest.fixture
def df_train1(df):
    return df.loc[df.cv.ne(1)].reset_index(drop=True)


@pytest.fixture
def df_train2_ts(df):
    return df.loc[df.cv.eq(1) | df.cv.eq(-1)].reset_index(drop=True)


@pytest.fixture
def df_test2_ts(df):
    return df.loc[df.cv.ge(2)].reset_index(drop=True)


@pytest.fixture
def df_train2(df):
    return df.loc[df.cv.ne(2)].reset_index(drop=True)


@pytest.fixture
def df_train3(df):
    return df.loc[df.cv.ne(3)].reset_index(drop=True)


@pytest.fixture
def df_test1(df):
    return df.loc[df.cv.eq(1)].reset_index(drop=True)


@pytest.fixture
def df_test2(df):
    return df.loc[df.cv.eq(2)].reset_index(drop=True)


@pytest.fixture
def df_test3(df):
    return df.loc[df.cv.eq(3)].reset_index(drop=True)


def test_val_set(df, df_val, df_train):
    # Check that the validation set is the same as the one passed in
    (
        assert_frame_equal(
            df_val.reset_index(drop=True), df.loc[df.cv.eq(-1)].reset_index(drop=True)
        ),
        "Validation set is not the same as the one passed in",
    )
    (
        assert_frame_equal(
            df_train.reset_index(drop=True), df.loc[df.cv.ne(-1)].reset_index(drop=True)
        ),
        "Training set is not the same as the one passed in",
    )

    # Get X_train, y_train, X_test, y_test
    X_train = df_train[["feature_col"]]
    y_train = df_train["target_col"]
    X_test = df_val[["feature_col"]]
    y_test = df_val["target_col"]

    result = time_series_validation_filter(
        df=df.query("cv != -1"),
        df_val=df_val,
        fold=None,
        fold_col="cv",
        feature_col="feature_col",
        target_col="target_col",
        time_series_validation=False,
    )
    (
        assert_frame_equal(
            X_train.reset_index(drop=True),
            result[0].reset_index(drop=True),
            f"X_train: {X_train} is not the same as the one returned: {result[0]}",
        )
    )
    (
        assert_series_equal(
            y_train,
            result[1],
            f"y_train: {y_train} is not the same as the one returned: {result[1]}",
        )
    )
    (
        assert_frame_equal(
            X_test.reset_index(drop=True),
            result[2].reset_index(drop=True),
            f"X_test: {X_test} is not the same as the one returned: {result[2]}",
        )
    )
    (
        assert_series_equal(
            y_test,
            result[3],
            f"y_test: {y_test} is not the same as the one returned: {result[3]}",
        )
    )


def test_cv1_false(
    df,
    df_train1,
    df_test1,
    # df_train2,
    # df_test2,
    # df_train3,
    # df_test3,
):
    train = df_train1
    test = df_test1
    print(type(df))
    result = time_series_validation_filter(
        df=df,
        df_val=None,
        fold=1,
        fold_col="cv",
        feature_col="feature_col",
        target_col="target_col",
        time_series_validation=False,
    )
    (
        assert_frame_equal(
            train[["feature_col"]].reset_index(drop=True),
            result[0].reset_index(drop=True),
            f"X_train: {train[['feature_col']]} is not the same as the one returned: {result[0]}",
        )
    )
    (
        assert_series_equal(
            train["target_col"],
            result[1].reset_index(drop=True),
            f"y_train: {train['target_col']} is not the same as the one returned: {result[1]}",
        )
    )
    (
        assert_frame_equal(
            test[["feature_col"]].reset_index(drop=True),
            result[2].reset_index(drop=True),
            f"X_test: {test[['feature_col']]} is not the same as the one returned: {result[2]}",
        )
    )
    (
        assert_series_equal(
            test["target_col"],
            result[3].reset_index(drop=True),
            f"y_test: {test['target_col']} is not the same as the one returned: {result[3]}",
        )
    )


def test_cv2_false(
    df,
    # df_train1,
    # df_test1,
    df_train2,
    df_test2,
    # df_train3,
    # df_test3,
):
    train = df_train2
    test = df_test2
    print(type(df))
    result = time_series_validation_filter(
        df=df,
        df_val=None,
        fold=2,
        fold_col="cv",
        feature_col="feature_col",
        target_col="target_col",
        time_series_validation=False,
    )
    (
        assert_frame_equal(
            train[["feature_col"]].reset_index(drop=True),
            result[0].reset_index(drop=True),
            f"X_train: {train[['feature_col']]} is not the same as the one returned: {result[0]}",
        )
    )
    (
        assert_series_equal(
            train["target_col"],
            result[1].reset_index(drop=True),
            f"y_train: {train['target_col']} is not the same as the one returned: {result[1]}",
        )
    )
    (
        assert_frame_equal(
            test[["feature_col"]].reset_index(drop=True),
            result[2].reset_index(drop=True),
            f"X_test: {test[['feature_col']]} is not the same as the one returned: {result[2]}",
        )
    )
    (
        assert_series_equal(
            test["target_col"],
            result[3].reset_index(drop=True),
            f"y_test: {test['target_col']} is not the same as the one returned: {result[3]}",
        )
    )


def test_cv2_true(
    df,
    df_train2_ts,
    df_test2_ts,
):
    train = df_train2_ts
    test = df_test2_ts
    print(type(df))
    result = time_series_validation_filter(
        df=df,
        df_val=None,
        fold=2,
        fold_col="cv",
        feature_col="feature_col",
        target_col="target_col",
        time_series_validation=True,
    )
    (
        assert_frame_equal(
            train[["feature_col"]].reset_index(drop=True),
            result[0].reset_index(drop=True),
            f"X_train: {train[['feature_col']]} is not the same as the one returned: {result[0]}",
        )
    )
    (
        assert_series_equal(
            train["target_col"],
            result[1].reset_index(drop=True),
            f"y_train: {train['target_col']} is not the same as the one returned: {result[1]}",
        )
    )
    (
        assert_frame_equal(
            test[["feature_col"]].reset_index(drop=True),
            result[2].reset_index(drop=True),
            f"X_test: {test[['feature_col']]} is not the same as the one returned: {result[2]}",
        )
    )
    (
        assert_series_equal(
            test["target_col"],
            result[3].reset_index(drop=True),
            f"y_test: {test['target_col']} is not the same as the one returned: {result[3]}",
        )
    )
