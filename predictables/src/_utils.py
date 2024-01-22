"""
Utility functions for PredicTables
"""

from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

output_map = {
    "dataframe": "dataframe",
    "df": "dataframe",
    "data": "dataframe",
    "table": "dataframe",
    "series": "series",
    "ser": "series",
    "s": "series",
    "lazyframe": "lazyframe",
    "lazy": "lazyframe",
    "lf": "lazyframe",
    "lazy_frame": "lazyframe",
}


def _to_numpy(
    data: Union[
        pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list
    ]
) -> np.ndarray:
    """
    Converts the data to a numpy array. Handles both pandas and polars dataframes and
    Series, as well as polars lazy frames, numpy arrays, and lists.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list]
        The data to be converted.

    Returns
    -------
    np.ndarray
        The converted data.
    """
    # Check each input type step by step
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, pd.Series):
        return data.to_numpy()
    elif isinstance(data, pl.DataFrame):
        return data.to_numpy()
    elif isinstance(data, pl.Series):
        return data.to_numpy()
    elif isinstance(data, pl.LazyFrame):
        return data.collect().to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError(
            f"Input data type {type(data)} not supported. \n\
Please use one of the following types: \n\
    - pandas.DataFrame \n\
    - pandas.Series \n\
    - polars.DataFrame \n\
    - polars.Series \n\
    - polars.LazyFrame \n\
    - numpy.ndarray \n\
    - list"
        )


def _to_polars(
    data: Union[
        pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list
    ],
    to: str = "dataframe",
) -> Union[pl.DataFrame, pl.Series, pl.LazyFrame]:
    """
    Converts the data to a polars dataframe, series, or lazy frame depending on the
    `to` parameter. Handles both pandas and polars dataframes and Series, as well as
    polars lazy frames, numpy arrays, and lists.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list]
        The data to be converted.
    to : str, optional
        The type to convert the data to. The string is preprocessed by passing it through
        the `output_map` dictionary. The default is 'dataframe'.

    Returns
    -------
    Union[pl.DataFrame, pl.Series, pl.LazyFrame]
        The converted data.
    """
    # Check output type
    to = output_map[to.lower()]

    # Check each input type step by step
    if isinstance(data, pd.DataFrame):
        if to == "dataframe":
            return pl.from_pandas(data)
        elif to == "lazyframe":
            return pl.from_pandas(data).lazy()
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pd.Series):
        if to == "dataframe":
            return pl.from_pandas(data.to_frame())
        elif to == "series":
            return pl.from_pandas(data)
        elif to == "lazyframe":
            return pl.from_pandas(data.to_frame()).lazy()
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pl.DataFrame):
        if to == "dataframe":
            return data
        elif to == "lazyframe":
            return data.lazy()
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pl.Series):
        if to == "dataframe":
            return data.to_frame()
        elif to == "series":
            return data
        elif to == "lazyframe":
            return data.to_frame().lazy()
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pl.LazyFrame):
        if to == "dataframe":
            return data.collect()
        elif to == "lazyframe":
            return data
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, np.ndarray):
        if to == "dataframe":
            return pl.from_numpy(data)
        elif to == "lazyframe":
            return pl.from_numpy(data).lazy()
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, list):
        if to == "dataframe":
            return pl.DataFrame(data)
        elif to == "lazyframe":
            return pl.DataFrame(data).lazy()
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    else:
        raise TypeError(
            f"Input data type {type(data)} not supported. \n\
Please use one of the following types: \n\
    - pandas.DataFrame \n\
    - pandas.Series \n\
    - polars.DataFrame \n\
    - polars.Series \n\
    - polars.LazyFrame \n\
    - numpy.ndarray \n\
    - list"
        )


def _to_pandas(
    data: Union[
        pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list
    ],
    to: str = "dataframe",
) -> Union[pd.DataFrame, pd.Series]:
    """
    Converts the data to a pandas dataframe or series depending on the
    `to` parameter. Handles both pandas and polars dataframes and Series, as well as
    polars lazy frames, numpy arrays, and lists.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list]
        The data to be converted.
    to : str, optional
        The type to convert the data to. The string is preprocessed by passing it through
        the `output_map` dictionary. The default is 'dataframe'. Will raise an error if
        `to` maps to 'lazyframe'.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        The converted data.
    """
    # Check output type
    to = output_map[to.lower()]

    # Check each input type step by step
    if isinstance(data, pd.DataFrame):
        if to == "dataframe":
            return data
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pd.Series):
        if to == "dataframe":
            return data.to_frame()
        elif to == "series":
            return data
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pl.DataFrame):
        if to == "dataframe":
            return data.to_pandas()
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pl.Series):
        if to == "dataframe":
            return data.to_pandas().to_frame()
        elif to == "series":
            return data.to_pandas()
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, pl.LazyFrame):
        if to == "dataframe":
            return data.collect().to_pandas()
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, np.ndarray):
        if to == "dataframe":
            return pd.DataFrame(data)
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    elif isinstance(data, list):
        if to == "dataframe":
            return pd.DataFrame(data)
        elif to == "lazyframe":
            raise ValueError(
                "Cannot convert to pandas lazy frame. \
There is no such thing as a lazy pandas frame."
            )
        else:
            raise ValueError(
                f"Output type {to} not supported for input type {type(data)}."
            )
    else:
        raise TypeError(
            f"Input data type {type(data)} not supported. \n\
Please use one of the following types: \n\
    - pandas.DataFrame \n\
    - pandas.Series \n\
    - polars.DataFrame \n\
    - polars.Series \n\
    - polars.LazyFrame \n\
    - numpy.ndarray \n\
    - list"
        )


def _select_binary_columns(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], missing_col: str = "999"
) -> list:
    """
    Returns the binary-coded columns from the data.

    Binary-coded columns are defined to be categorical, with at
    most three categories:
        1. "0"
        2. "1"
        3. "-999" / "missing"

    If there are more than three categories, the column is not binary.
    If there are three or fewer three categories, and the categories are
    not some subset of "0", "1", and "-999" (or another value indicating
    missingness), the column is not binary.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The data to be analyzed.
    missing_col : str
        The name of the column that indicates missingness. This column
        denotes the third possible category of a binary column (the other
        two being "0" and "1").

    Returns
    -------
    list
        The names of the binary columns.
    """
    # Convert to polars lazy frame if necessary
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data).lazy()
    elif isinstance(data, pl.DataFrame):
        data = data.lazy()

    # Select all categorical columns
    categorical = data.select(cs.categorical() | cs.string())

    # Drop columns that have > 3 categories
    binary = categorical.select(
        [
            col
            for col in categorical.columns
            if categorical.select([col]).unique().collect().shape[0] <= 3
        ]
    )

    # Drop any other columns that have something besides "0", "1", and missing_col as categories
    binary = binary.select(
        [
            col
            for col in binary.columns
            if set(
                binary.select([col])
                .unique()
                .collect()
                .select("gt_mean")
                .unique()
                .to_series()
            )
            <= {"0", "1", missing_col}
        ]
    )

    # Return the column names
    return binary.columns
