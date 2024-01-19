import polars as pl

from predictables.util import to_pl_lf

from .BaseModel import Model


class SingleUnivariate(Model):
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
        self.unique_folds = (
            to_pl_lf(self.df)
            .select(self.fold_col)
            .unique()
            .collect()
            .to_pandas()[self.fold_col]
            .values
        )
        self.cv = {}
        for fold in self.unique_folds:
            self.cv[fold] = Model(
                self.df,
                fold=fold,
                fold_col=self.fold_col,
                feature_col=self.feature_col,
                target_col=self.target_col,
            )
