import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


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
    corr = compute_feature_correlations(lf).filter(
        pl.col('correlation') > threshold
    ).select([
        pl.col('col1'),
        pl.col('col2')
    ]).collect().to_pandas()

# def evaluate_model_with_removal(
#     model, data, target, cv_generator, features, feature_to_remove
# ):
#     features_without = [f for f in features if f != feature_to_remove]
#     aucs = []
#     for train_idx, test_idx in cv_generator.split(data):
#         model.fit(data.iloc[train_idx][features_without], target.iloc[train_idx])
#         pred_probs = model.predict_proba(data.iloc[test_idx][features_without])[:, 1]
#         aucs.append(roc_auc_score(target.iloc[test_idx], pred_probs))
#     return np.mean(aucs), np.std(aucs)


# def decide_feature_removal_based_on_auc(
#     auc_info1, auc_info2, feature1, feature2, current_features
# ):
#     mean_auc1, std_auc1 = auc_info1
#     mean_auc2, std_auc2 = auc_info2
#     if mean_auc1 > mean_auc2:
#         current_features.remove(feature2)
#     elif mean_auc2 > mean_auc1:
#         current_features.remove(feature1)


# # Create the dataset

# np.random.seed(42)  # For reproducibility

# # Base feature
# base = np.random.normal(0, 1, 100)

# # Highly correlated feature (~90%)
# high_corr = base * 0.9 + np.random.normal(0, 0.3, 100)

# # Medium correlated feature (~55-60%)
# medium_corr = base * 0.6 + np.random.normal(0, 0.8, 100)

# # Weakly correlated feature (~20-25%)
# weak_corr = base * 0.25 + np.random.normal(0, 1, 100)

# # Uncorrelated feature
# uncorrelated = np.random.normal(0, 1, 100)

# data = pd.DataFrame(
#     {
#         "Base": base,
#         "High_Corr": high_corr,
#         "Medium_Corr": medium_corr,
#         "Weak_Corr": weak_corr,
#         "Uncorrelated": uncorrelated,
#     }
# )

# # Target variable (binary outcome)
# target = np.random.randint(0, 2, size=100)

# # Set up model and cross-validator
# model = LogisticRegression(solver="liblinear")
# cv = KFold(n_splits=5, shuffle=True, random_state=42)

# # Full integration test with expectations


# def full_integration_test_with_expectations(data, target, model, cv):
#     current_features = data.columns.tolist()
#     correlations = compute_feature_correlations(data)
#     print("Initial Correlations:\n", correlations)

#     initial_feature_set = set(current_features)

#     for i in range(len(current_features) - 1):  # Iteratively remove features
#         for j in range(i + 1, len(current_features)):
#             feature1 = current_features[i]
#             feature2 = current_features[j]
#             if feature1 not in current_features or feature2 not in current_features:
#                 continue
#             auc_info1 = evaluate_model_with_removal(
#                 model, data, target, cv, current_features, feature2
#             )
#             auc_info2 = evaluate_model_with_removal(
#                 model, data, target, cv, current_features, feature1
#             )
#             decide_feature_removal_based_on_auc(
#                 auc_info1, auc_info2, feature1, feature2, current_features
#             )

#     final_features = set(current_features)
#     expected_removals = {"High_Corr", "Medium_Corr"}
#     expected_retentions = {"Weak_Corr", "Uncorrelated"}

#     removed_as_expected = (
#         expected_removals.intersection(initial_feature_set) - final_features
#     )
#     retained_as_expected = expected_retentions.intersection(final_features)

#     print("Final features: ", current_features)
#     assert (
#         removed_as_expected == expected_removals
#     ), f"Features {expected_removals - removed_as_expected} were not removed as expected."
#     assert (
#         retained_as_expected == expected_retentions
#     ), f"Features {expected_retentions - retained_as_expected} were not retained as expected."


# # Execute the full integration test with expectations
# full_integration_test_with_expectations(data, target, model, cv)
