from dataclasses import dataclass

import pandas as pd

from ._BaseLookup import BaseLookup


@dataclass
class DateLookup(BaseLookup):
    super().__init__()

    df: pd.DataFrame = None
    name: str = "date"
    description: str = "A lookup table for dates"
    id_column: str = "date_id"
