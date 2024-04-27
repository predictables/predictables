import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier



def backward_stepwise_feature_selection(model: CatBoostClassifier, threshold: float = 0.5) -> pl.LazyFrame:
    """Select a subset of the current features by iteratively removing highly-correlated features that do not significantly impact the model."""
    gen = generate_X_y()
    X_train, y_train, X_test, y_test = next(gen)
    current_features = X_train.columns
    corr_pairs = identify_highly_correlated_pairs(X_train, threshold)

    for col1, col2 in corr_pairs:
        current_model, model_ex_1, model_ex_2 = fit_models(model, col1, col2)
        current_auc, ex1_auc, ex2_auc = roc_auc_eval(
            current_model, model_ex_1, model_ex_2, col1, col2
        )

        # Test whichever column has the highest auc
        if mean(ex1_auc) > mean(ex2_auc):

            # If the current AUC is within a SD of mean(ex1_auc)
            # drop the column
            if mean(current_auc) 



def fit_models(model: CatBoostClassifier, col1: str, col2: str) -> tuple[list[CatBoostClassifier], list[CatBoostClassifier], list[CatBoostClassifier]]:
    gen_cur, gen1, gen2 = generate_X_y(), generate_X_y(), generate_X_y()

    cur_models = [
        model.fit(
            X_train.collect().to_numpy(),
            y_train.to_numpy().ravel()
        )
        for X_train, y_train, _, _
        in gen_cur
    ]

    ex1_models = [
        model.fit(
            X_train.collect().to_numpy(),
            y_train.to_numpy().ravel()
        )
        for X_train, y_train, _, _
        in gen1
    ]

    ex2_models = [
        model.fit(
            X_train.collect().to_numpy(),
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
