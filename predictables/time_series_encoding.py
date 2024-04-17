import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def synthetic_dataframe():
    np.random.seed(42)
    data = {
        'index': np.arange(200),
        'logit[MEAN_ENCODED_feature1_30]': np.random.rand(200) * 100,
        'logit[MEAN_ENCODED_feature1_60]': np.random.rand(200) * 100,
        'logit[MEAN_ENCODED_feature1_90]': np.random.rand(200) * 100,
        'logit[MEAN_ENCODED_feature2_30]': np.random.rand(200) * 200,
        'logit[MEAN_ENCODED_feature2_60]': np.random.rand(200) * 200,
        'logit[MEAN_ENCODED_feature2_90]': np.random.rand(200) * 200,
    }
    df = pd.DataFrame(data)
    return df

def load_and_preprocess_data(filepath):
    df = pd.read_parquet(filepath)
    logit_columns = [col for col in df.columns if col.startswith('logit')]
    columns_to_keep = ['index'] + logit_columns
    df_selected = df[columns_to_keep]
    return df_selected

def extract_features_and_lags(df):
    features = []
    lags = []
    for col in df.columns:
        if col.startswith('logit'):
            parts = col.split('_')
            feature = '_'.join(parts[2:-1])
            lag = int(parts[-1].replace(']', ''))
            features.append(feature)
            lags.append(lag)
            df.rename(columns={col: f'{feature}_{int(lag/30)}'}, inplace=True)
    return df, features, lags

def time_series_cv_corrected(df, features, max_lag, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for p in range(1, max_lag + 1):
        mse_scores = []
        for train_index, test_index in tscv.split(df):
            train_df, test_df = df.iloc[train_index], df.iloc[test_index]
            feature_cols = [f'{feature}_{lag}' for feature in features for lag in range(1, max_lag + 1) if f'{feature}_{lag}' in df.columns]
            X_train = train_df[feature_cols].values
            X_test = test_df[feature_cols].values
            if f'{features[0]}_{p}' in df.columns:
                y_train = train_df[f'{features[0]}_{p}'].values
                y_test = test_df[f'{features[0]}_{p}'].values
            else:
                continue
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mse_scores.append(mse)
        if mse_scores:
            avg_mse = np.mean(mse_scores)
            results.append((p, avg_mse))
    best_p = min(results, key=lambda x: x[1])[0] if results else None
    return results, best_p

# Main execution
df_synthetic = synthetic_dataframe()
df_transformed, features, _ = extract_features_and_lags(df_synthetic.copy())
results, best_p = time_series_cv_corrected(df_transformed, features, max_lag=3, n_splits=3)
print("Results (Lag, MSE):", results)
print("Best lag period based on MSE:", best_p)
df_transformed.head()