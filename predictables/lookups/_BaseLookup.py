from __future__ import annotations
from dataclasses import dataclass

import pandas as pd


@dataclass
class BaseLookup:
    name: str | None = None
    description: str | None = None
    id_column: str | None = None
    df: pd.DataFrame | None = None

    def join_id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(self.df, on=self.id_column, how="left")

    def join_name(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(self.df, on=self.name, how="left")
