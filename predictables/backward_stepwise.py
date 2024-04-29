"""Perform backward stepwise feature selection by recursively eliminating features from a model based on their correlation and impact on predictive performance, evaluated through AUC metrics.

    The process involves:
    1. Identifying pairs of features with a Pearson correlation coefficient above a specified threshold (default is 0.5).
    2. For each correlated pair, the model's performance is evaluated three times: using all features, minus the first feature, and minus the second feature.
    3. The feature whose removal causes the least decrease in AUC (within one standard deviation of the AUC when all features are used) is eliminated. This standard deviation acts as a threshold to ensure that only features whose absence does not statistically degrade the model's performance are removed.
    4. This iterative process continues until all pairs of correlated features have been evaluated, refining the feature set to improve model simplicity without significantly reducing performance.

    Args:
    - model (SKClassifier): The pre-trained model whose features are to be evaluated.
    - threshold (float): The correlation threshold above which feature pairs are considered for elimination.

    Returns:
    - pl.LazyFrame: The dataset with the reduced set of features after removing highly correlated and less impactful ones.
"""
import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from predictables.util import SKClassifier # duck type indicating that the model is an sklearn classifier implementing the fit and predict methods 


def backward_stepwise_feature_selection(X: pl.LazyFrame, y: np.ndarray, model: SKClassifier, threshold: float = 0.5) -> pl.LazyFrame:
    """Select a subset of the current features by iteratively removing highly-correlated features that do not significantly impact the model."""
    gen = generate_X_y(X, y)
    X_train, y_train, X_test, y_test = next(gen)
    current_features = X_train.columns
    
    columns_to_drop = []
    
    corr_pairs = identify_highly_correlated_pairs(X_train, threshold)

    for col1, col2 in corr_pairs:
        current_auc, ex1_auc, ex2_auc = evaluate_single_model(model, col1, col2, columns_to_drop)
        
        col_to_drop = evaluate_what_if_any_column_to_drop(current_auc, ex1_auc, ex2_auc)

        if col_to_drop > 0:
            columns_to_drop.append(drop_column_from_data(col_to_drop, col1, col2))
            
    return X_train.drop(columns_to_drop)
            

def generate_X_y(X: pl.LazyFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Yield training and testing splits for time series cross validation."""
    for i in range(5, 11):
        train_idx = X.with_row_index().filter(
            pl.col("fold") < i+1
        ).select(
            pl.col("index")
        ).collect().to_numpy()
        
        val_idx = X.with_row_index().filter(
            pl.col("fold") == i+1
        ).select(
            pl.col("index")
        ).collect().to_numpy()
        
        dfx = X.collect().to_pandas()
        
        yield dfx.iloc[train_idx], y[train_idx], dfx.iloc[val_idx], y[val_idx]
    
def evaluate_what_if_any_column_to_drop(current_auc, ex1_auc, ex2_auc) -> int:
    """Return 0, 1, or 2 to indicate you should drop no column, column 1, or column 2, respectively.
    
    If the mean AUC from the model fit without either column 1 or 2 is greater than the mean AUC
    of the current model less one standard deviation, make the new current
    model be the one with the highest mean AUC by returning either 1 or 2 depending on which was
    higher. If both smaller models are worse than the current, do not make any adjustment to the
    current model and continue to the next pair-return 0 to indicate this.
    """
    # Test whichever column has the highest auc
    if max(np.mean(ex1_auc), np.mean(ex2_auc)) > np.mean(current_auc) - np.std(current_auc):
        return 1 if np.mean(ex1_auc) < np.mean(ex2_auc) else 2
    else:
        return 0
            

def evaluate_single_model(model: SKClassifier, col1: str, col2: str, columns_to_drop: list[str]):
    current_model, model_ex_1, model_ex_2 = fit_models(model, col1, col2, columns_to_drop)
    current_auc, ex1_auc, ex2_auc = roc_auc_eval(
        current_model, model_ex_1, model_ex_2, col1, col2
    )
    
    return current_auc, ex1_auc, ex2_auc

def fit_models(model: SKClassifier, col1: str, col2: str, columns_to_drop: list[str]) -> tuple[list[SKClassifier], list[SKClassifier], list[SKClassifier]]:
    gen_cur, gen1, gen2 = generate_X_y(), generate_X_y(), generate_X_y()

    cur_models = [
        model.fit(
            X_train.drop(columns_to_drop).collect().to_numpy(),
            y_train.to_numpy().ravel()
        )
        for X_train, y_train, _, _
        in gen_cur
    ]

    ex1_models = [
        model.fit(
            X_train.drop(columns_to_drop).collect().to_numpy(),
            y_train.to_numpy().ravel()
        )
        for X_train, y_train, _, _
        in gen1
    ]

    ex2_models = [
        model.fit(
            X_train.drop(columns_to_drop).collect().to_numpy(),
            y_train.to_numpy().ravel()
        )
        for X_train, y_train, _, _
        in gen2
    ]

    return cur_models, ex1_models, ex2_models

def compute_feature_correlations(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute the correlation between all features in the dataset."""
    cols = lf.columns
    return (
        lf.select(
            [
                pl.corr(cols[i], cols[j]).alias(f"{cols[i]}_corr_{cols[j]}")
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
            ]
        )
        .collect()
        .transpose(include_header=True)
        .lazy()
        .with_columns(
            [
                pl.col("column").str.split("_corr_").list.get(0).alias("col1"),
                pl.col("column").str.split("_corr_").list.get(1).alias("col2"),
                pl.col("column_0").abs().alias("correlation"),
            ]
        )
        .drop("column_0")
        .drop("column")
        .filter(pl.col("col1") != pl.col("col2"))
        .sort("correlation", descending=True)
    )


def identify_highly_correlated_pairs(
    lf: pl.LazyFrame, threshold: float = 0.5
) -> list[tuple[str, str]]:
    """Return highly-correlated pairs of columns.
    
    Here 'highly-correlated' means that the absolute value of their 
    Pearson correlation coefficient is > 0.5. While this might seem
    like a low threshold, keep in mind that each pair of columns will be
    emprically tested, so the decision was made to keep the threshold
    fairly low to ensure that we don't bother with obviously weakly-
    correlated pairs, but also ensure that all marginal cases are 
    actually tested.
    """   
    corr = compute_feature_correlations(lf).filter(
        pl.col('correlation') > threshold
    ).select([
        pl.col('col1'),
        pl.col('col2')
    ]).collect().to_pandas()
