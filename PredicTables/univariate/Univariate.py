import pandas as pd
import polars as pl

from .SingleUnivariate import SingleUnivariate


def get_col(self, col):
    attributes = [getattr(self.cv[fold], col) for fold in self.unique_folds]
    sd = pd.Series(attributes).std()

    out = attributes + [getattr(self, col)] + [sd]
    return out


class Univariate(SingleUnivariate):
    def __init__(
        self,
        df: pl.LazyFrame,
        fold_col: str = "cv",
        feature_col: str = None,
        target_col: str = None,
    ):
        super().__init__(
            df, fold_col=fold_col, feature_col=feature_col, target_col=target_col
        )
        self.results = pd.DataFrame(
            index=sorted(self.unique_folds.tolist()) + ["mean", "std"]
        )
        self.results.index.name = "fold"
        for attribute in [
            "coef",
            "pvalues",
            "se",
            "lower_ci",
            "upper_ci",
            "acc_train",
            "acc_test",
            "auc_train",
            "auc_test",
            "f1_train",
            "f1_test",
            "precision_train",
            "precision_test",
            "recall_train",
            "recall_test",
            "mcc_train",
            "mcc_test",
            "logloss_train",
            "logloss_test",
            "auc_train",
            "auc_test",
        ]:
            self.results[attribute] = get_col(self, attribute)
