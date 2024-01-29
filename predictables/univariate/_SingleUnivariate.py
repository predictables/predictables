from typing import Optional, Union

import pandas as pd

from predictables.util import get_unique

from ._BaseModel import Model


class SingleUnivariate(Model):
    unique_folds: Union[pd.Series, list]
    cv_dict: dict

    def __init__(
        self,
        df: pd.DataFrame,
        fold_col: str = "cv",
        feature_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ):
        super().__init__(
            df, fold_col=fold_col, feature_col=feature_col, target_col=target_col
        )
        self.unique_folds = get_unique(self.df.loc[:, self.fold_col])
        self.cv_dict = {}
        for fold in self.unique_folds:
            self.cv_dict[fold] = Model(
                self.df,
                fold_n=fold,
                fold_col=self.fold_col,
                feature_col=self.feature_col if self.feature_col is not None else None,
                target_col=self.target_col if self.target_col is not None else None,
            )
