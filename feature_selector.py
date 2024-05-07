"""Contains the code to use the feature selector."""
from __future__ import annotations
import polars as pl  # type: ignore
import polars.selectors as cs  # type: ignore
import numpy as np
from catboost import CatBoostClassifier  # type: ignore


from feature_selector_constants import (  # type: ignore
    MAIN_DATA_FILE,
    CV_FOLD_DATA_FILE,
    TARGET,
    TRAINING_DATA_FILE,
    COLS_TO_DROP,
    CATBOOST_HYPERPARAMETERS
)
from feature_selector_helpers import (  # type: ignore
    X_y_generator,
    get_df,
    get_X,
    get_y,
    next_gen_gen,
)


class FeatureSelector:
    def __init__(
        self,
        col1: str, 
        col2: str,
        X: pd.DataFrame = get_X(),
        y: pd.Series = get_y(),
        generator=lambda : next_gen_gen(get_X(), get_y(), CATBOOST_HYPERPARAMETERS),
        data_file: str = MAIN_DATA_FILE,
    ):

        self.col1 = col1
        self.col2 = col2

        self.X = X
        self.y = y

        self.data_file = data_file
        self.lf = pl.scan_parquet(self.data_file).drop(COLS_TO_DROP)
        self.generator = generator

        self.correlated_columns_generator = self.get_correlated_cols()

        self.before_auc: list[float] = []
        self.after_dropping_col1_aucs: list[float] = []
        self.after_dropping_col2_aucs: list[float] = []

        self.get_before_results()
        self.get_after_results()

        self.mean_before = np.mean(self.before_auc)
        self.sd_before = np.std(self.before_auc)
        self.mean_after1 = np.mean(self.after_dropping_col1_aucs)
        self.mean_after2 = np.mean(self.after_dropping_col2_aucs)

        self.col_to_drop:str = ''

        if np.round(self.mean_after1, 4) >= np.round(self.mean_after2, 4):
            if np.round(self.mean_after2) <= np.round(self.mean_before - self.sd_before, 4):
                self.col_to_drop = self.col2
            else:
                self.col_to_drop = 'none'

        else:
            if np.round(self.mean_after1) <= np.round(self.mean_before - self.sd_before, 4):
                self.col_to_drop = self.col1
            else:
                self.col_to_drop = 'none'

    def __call__(self):
        return self.col_to_drop

    def _fit_single_model(X_train, y_train, X_test, y_test, model):
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def _get_single_auc(X_train, y_train, X_test, y_test, model):
        y_pred = _fit_single_model(X_train, y_train, X_test, y_test, model)
        return roc_auc_score(y_test, y_pred)

    def get_before_results(self):
        gen = self.generator()
        for X_train, y_train, X_test, y_test, model in gen:
            self.before_auc.append(
                _get_single_auc(X_train, y_train, X_test, y_test, model)
            )

    def get_after_results(self):
        gen = self.generator()
        for X_train, y_train, X_test, y_test, model in gen:
            self.after_dropping_col1_aucs.append(
                _get_single_auc(X_train.drop([self.col1]), y_train, X_test.drop([self.col1]), y_test, model)
            )
            self.after_dropping_col2_aucs.append(
                _get_single_auc(X_train.drop([self.col2]), y_train, X_test.drop([self.col2]), y_test, model)
            )


    def get_correlated_cols(self, threshold: float = 0.5):
        cols = self.X.columns
        len_cols = len(cols)
        
        corr_lf = self.X.select([
            pl.corr(pl.col(cols[i]), pl.col(cols[j])).alias(f"{cols[i]}_corr_{cols[j]}")
            
            for i in range(len_cols)
            for j in range(i+1, len_cols)
        ])
        
        corr = corr_lf.collect().transpose(
            include_header=True,
            header_name="cols"
        ).lazy().select([
            pl.col('cols'),
            pl.col('column_0').alias('corr'),
        ]).with_columns([
            pl.col('corr').abs().alias('abs_corr'),
            pl.col('cols').str.split("_corr_").list.get(0).alias('col1'),
            pl.col('cols').str.split("_corr_").list.get(1).alias('col2')
        ]).sort(
            ['abs_corr',
            'col1',
            'col2'],
            descending=True
        ).filter(
            pl.col('corr').is_not_nan()
        ).filter(
            pl.col('abs_corr') > threshold
        ).select([
            pl.col('col1'),
            pl.col('col2'),
            pl.col('corr').round(5),
            pl.col('abs_corr').round(5)
        ]).collect()
        
        for row in corr.iter_rows():
            yield row
