# import pytest
# import pandas as pd
# from PredicTables.impute import ModelImputer

# @pytest.fixture
# def synthetic_data():
#     # X, y = generate_synthetic_data_cat(between_correlation=0.1)
#     # for col in [f"feature_{i}" for i in [10, 11, 12]]:
#     #     X[col] = X[col].astype(int).astype(str).astype("category")

#     # X = X[[f"feature_{i}" for i in range(10, 13)] + [f"feature_{i}" for i in range(10)]]

#     # Xm = X.copy()
#     # missing_indicator = np.random.binomial(1, 0.2, (Xm.shape[0], 3))
#     # missing_indicator = pd.DataFrame(
#     #     missing_indicator, columns=["feature_12", "feature_0", "feature_1"]
#     # )

#     # Xm.loc[missing_indicator["feature_12"] == 1, "feature_12"] = np.nan
#     # Xm.loc[missing_indicator["feature_0"] == 1, "feature_0"] = np.nan
#     # Xm.loc[missing_indicator["feature_1"] == 1, "feature_1"] = np.nan

#     # missing_mask = pd.DataFrame(missing_indicator == 1, columns=X.columns).fillna(False)

#     Xm = pd.DataFrame(
#         {
#             "feature_10": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             "feature_11": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             "feature_12": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             "feature_0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         }
#     )

#     missing_mask = pd.DataFrame(
#         {
#             "feature_10": [
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#             ],
#             "feature_11": [
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#             ],
#             "feature_12": [
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#             ],
#             "feature_0": [
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#             ],
#             "feature_1": [
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#                 False,
#             ],
#         }
#     )

#     return Xm, missing_mask

# @pytest.fixture
# def model_imputer(synthetic_data):
#     Xm, missing_mask = synthetic_data
#     mi = ModelImputer(Xm, missing_mask)
#     return mi

# @pytest.fixture
# def fitted_model_imputer(mi):
#     mi.fit_models()
#     return mi

# def test_ModelImputer_init(synthetic_data):
#     Xm, missing_mask = synthetic_data

#     # Drop columns if necessary:
#     if missing_mask.shape[1] > Xm.shape[1]:
#         missing_mask = missing_mask[Xm.columns.tolist()]
#     elif missing_mask.shape[1] < Xm.shape[1]:
#         Xm = Xm[missing_mask.columns.tolist()]
#     mi = ModelImputer(Xm, missing_mask)

#     # Check that data shapes persist
#     assert (
#         mi.df.shape == Xm.shape
#     ), f"mi.df.shape: {mi.df.shape}, Xm.shape: {Xm.shape} -- why aren't they the same?"
#     assert (
#         mi.missing_mask.shape == missing_mask.shape
#     ), f"mi.missing_mask.shape: {mi.missing_mask.shape}, missing_mask.shape: {missing_mask.shape} -- why aren't they the same?"

#     # Check that the missing_mask is full of boolean values
#     assert (
#         mi.missing_mask.dtypes == "bool"
#     ).all(), f"mi.missing_mask.dtypes: {mi.missing_mask.dtypes}\nmi.missing_mask.dtypes.value_counts: {mi.missing_mask.dtypes.value_counts()}\nmi.missing_mask.dtypes.unique: {mi.missing_mask.dtypes.unique()}\nmi.missing_mask.dtypes.nunique: {mi.missing_mask.dtypes.nunique()}\nmi.missing_mask.dtypes.value_counts: {mi.missing_mask.dtypes.value_counts()}\nmi.missing_mask.dtypes.value_counts.index: {mi.missing_mask.dtypes.value_counts().index}\nmi.missing_mask.dtypes.value_counts.index.dtype: {mi.missing_mask.dtypes.value_counts().index.dtype}\nmi.missing_mask.dtypes.value_counts.index.dtype.kind: {mi.missing_mask.dtypes.value_counts().index.dtype.kind}\nmi.missing_mask.dtypes.value_counts.index.dtype.name: {mi.missing_mask.dtypes.value_counts().index.dtype.name}\nmi.missing_mask.dtypes.value_counts.index.dtype.type: {mi.missing_mask.dtypes.value_counts().index.dtype.type}\nmi.missing_mask.dtypes.value_counts.index.dtype.kind: {mi.missing_mask.dtypes.value_counts().index.dtype.kind}\n"
