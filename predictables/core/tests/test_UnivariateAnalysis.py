import pytest
from unittest.mock import Mock, patch
import pandas as pd
from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
# from predictables.univariate import Univariate

@pytest.fixture
def df():
    return pd.read_parquet("cancerdf.parquet")

@pytest.fixture
def features(df):
    return df.columns.tolist()[:-2]

# Mocking the Univariate class
@pytest.fixture
def mock_univariate_class():
    with patch('predictables.univariate.Univariate', autospec=True) as mock:
        yield mock

@pytest.mark.parametrize("has_time_series_structure", [True, False])
def test_univariate_analysis_constructor(df, features, mock_univariate_class, has_time_series_structure):
    model_name = "TestModel"
    ua = UnivariateAnalysis(
        model_name=model_name,
        df_train=df,
        df_val=df,
        target_column_name='target',
        feature_column_names=features,
        cv_folds=df['cv'],
        has_time_series_structure=has_time_series_structure
    )
    # Test the basic attribute assignment
    assert ua.model_name == model_name
    assert ua.df.equals(df)
    assert ua.df_val.equals(df)
    assert ua.target_column_name == 'target'
    assert ua.feature_column_names == features
    assert ua.has_time_series_structure == has_time_series_structure
    assert len(ua._feature_list) == len(features)
    
    # Verify that Univariate instances were created for each feature
    mock_univariate_class.assert_called()

