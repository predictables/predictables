from dataclasses import dataclass

import pandas as pd


@dataclass
class BaseLookup:
    name: str = None
    description: str = None
    id_column: str = None
    df: pd.DataFrame = None

    def join_id(self, df):
        return df.merge(self.df, on=self.id_column, how="left")

    def join_name(self, df):
        return df.merge(self.df, on=self.name, how="left")
