from enum import Enum

import pandas as pd
import polars as pl


class DataType(Enum):
    """
    DataType enum

    This enum is used to represent the data type of a column in a table, and
    to ensure consistency across the different projects that comprise PredicTables.

    The enum is defined in this file, and is imported into the other projects as needed.

    The enum recognizes the following data types:
        - CONTINUOUS
        - CATEGORICAL
        - BINARY
        - DATE
        - OTHER

    CONTINUOUS is used for columns that contain continuous numerical data, such as
        - age
        - height
        - weight
        - temperature
        - etc.
    Note that CONTINUOUS does not include columns that contain categrories that
    have been encoded as numbers, such as
        - zip codes
        - phone numbers
        - policy type
        - etc.

    CATEGORICAL is used for columns that contain categorical data, such as
        - gender
        - marital status
        - etc.
    Note that CATEGORICAL does not include columns that contain categrories that
    have exactly two values, such as
        - yes/no
        - true/false
        - etc.
    but does include columns that contain categrories that have been encoded as numbers, such as
        - zip codes
        - phone numbers
        - policy type
        - etc.

    BINARY is used for columns that contain binary data, such as
        - yes/no
        - true/false
        - 0/1
        - -1/1
        - etc.

    DATE is used for columns that contain date data, such as
        - date of birth
        - date of death
        - date of policy issue
        - etc.

    OTHER is used for columns that contain data that does not fit into any of
    the above categories, or that makes no sense to categorize, and more
    like a string, such as
        - name
        - address
        - etc.

    The enum codes each data type as an integer, which is used in the

    """

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    DATE = "date"
    OTHER = "other"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, DataType):
            return self.name == other.name
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def is_continuous(self):
        return self == DataType.CONTINUOUS

    def is_categorical(self):
        return (self == DataType.CATEGORICAL) or (self == DataType.BINARY)

    def is_binary(self):
        return self == DataType.BINARY

    def is_date(self):
        return self == DataType.DATE

    def is_other(self):
        return self == DataType.OTHER

    def get_polars(self):
        if self == DataType.CONTINUOUS:
            return pl.Float64
        elif self == DataType.CATEGORICAL:
            return pl.Utf8
        elif self == DataType.BINARY:
            return pl.Boolean
        elif self == DataType.DATE:
            return pl.Date
        elif self == DataType.OTHER:
            return pl.Utf8
        else:
            raise ValueError(f"Unknown data type: {self}")

    def get_pandas(self):
        if self == DataType.CONTINUOUS:
            return pd.Float64
        elif self == DataType.CATEGORICAL:
            return pd.StringDtype()
        elif self == DataType.BINARY:
            return pd.BooleanDtype()
        elif self == DataType.DATE:
            return pd.DatetimeTZDtype()
        elif self == DataType.OTHER:
            return pd.StringDtype()
        else:
            raise ValueError(f"Unknown data type: {self}")

    @staticmethod
    def from_str(s: str) -> "DataType":
        if s == "continuous":
            return DataType.CONTINUOUS
        elif s == "categorical":
            return DataType.CATEGORICAL
        elif s == "binary":
            return DataType.BINARY
        elif s == "date":
            return DataType.DATE
        elif s == "other":
            return DataType.OTHER
        else:
            raise ValueError(f"Unknown data type: {s}")

    @staticmethod
    def from_pandas_series(col: pd.Series) -> "DataType":
        dtype = str(col.dtype)
        unique = col.drop_duplicates().sort_values()
        n_unique = len(unique)
        numeric_dtypes = [
            "int64",
            "int32",
            "int16",
            "int8",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
            "float64",
            "float32",
        ]

        categorical_dtypes = ["object", "string", "str", "category", "boolean", "bool"]

        date_dtypes = [
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, tz]",
            "datetime64[ms]",
        ]

        max_diff_eq_1 = False
        if dtype in numeric_dtypes:
            if n_unique > 1:
                max_diff = unique.diff().dropna().max()
                max_diff_eq_1 = max_diff == 1

        if dtype in numeric_dtypes:
            if n_unique <= 2:
                return DataType.BINARY
            elif max_diff_eq_1:
                return DataType.CATEGORICAL
            else:
                return DataType.CONTINUOUS
        elif dtype in categorical_dtypes:
            if n_unique <= 2:
                return DataType.BINARY
            else:
                return DataType.CATEGORICAL
        elif dtype in date_dtypes:
            return DataType.DATE
        else:
            raise ValueError(f"Unknown data type: {dtype}")

    @staticmethod
    def from_polars_series(col: pl.Series) -> "DataType":
        dtype = str(col.dtype)
        unique = col.unique().sort()
        n_unique = len(unique)
        numeric_dtypes = [
            "Int64",
            "Int32",
            "Int16",
            "Int8",
            "UInt64",
            "UInt32",
            "UInt16",
            "UInt8",
            "Float64",
            "Float32",
        ]

        categorical_dtypes = [
            "Object",
            "String",
            "Utf8",
            "Boolean",
            "Bool",
            "Category",
            "Categorical",
        ]

        date_dtypes = [
            "Date",
            "Datetime",
            "Time",
            "Duration",
            "NaiveDateTime",
            "DateTime",
        ]

        max_diff_eq_1 = False
        if dtype in numeric_dtypes:
            if n_unique > 1:
                max_diff = unique.diff().drop_nans().max()
                max_diff_eq_1 = max_diff == 1

        if dtype in numeric_dtypes:
            if n_unique <= 2:
                return DataType.BINARY
            elif max_diff_eq_1:
                return DataType.CATEGORICAL
            else:
                return DataType.CONTINUOUS
        elif dtype in categorical_dtypes:
            if n_unique <= 2:
                return DataType.BINARY
            else:
                return DataType.CATEGORICAL
        elif dtype in date_dtypes:
            return DataType.DATE
        else:
            raise ValueError(f"Unknown data type: {dtype}")
