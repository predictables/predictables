from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
from predictables.util import to_pd_df


class PredicTables:
    def __init__(
        self,
        df_train,
        df_val,
        df_test,
        target_column_name,
        cv_folds,
        has_time_series_structure,
    ):
        self.df_train = to_pd_df(df_train).assign(cv=cv_folds)
        self.df_val = to_pd_df(df_val)
        self.df_test = to_pd_df(df_test)

        self.target_column_name = target_column_name
        self.has_time_series_structure = has_time_series_structure

        self.feature_column_names = [
            col for col in self.df_train if col not in ["cv", self.target_column_name]
        ]

        self.ua = UnivariateAnalysis(
            self.df_train,
            self.target_column_name,
            self.feature_column_names,
            cv_folds,
            has_time_series_structure=self.has_time_series_structure,
        )
