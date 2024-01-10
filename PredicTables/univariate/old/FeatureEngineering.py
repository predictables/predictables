from typing import Union, List, Callable, Optional
import pandas as pd
import polars as pl
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import os
from itertools import product

from univariate._feature_engineering_params import (
    NON_NUMERIC_COLUMNS,
    NUMERIC_CATEGORICAL_COLUMNS,
    NON_CATEGORICAL_COLUMNS,
)


## functions used below:
def aggregate_features(
    df: Union[pl.LazyFrame, pl.DataFrame],
    num_cols: List[str],
    groupby_cols: List[str],
    func: Callable,
    func_name: str,
    agg_func: Optional[Union[Callable, List[Callable]]] = None,
    agg_func_name: Optional[Union[str, List[str]]] = None,
    include_relativities: bool = False,
) -> pl.LazyFrame:
    """
    Aggregate numeric features by categorical features, and join the aggregated features back to
    the original DataFrame. This is used to create features like mean encoding.

    Parameters
    ----------
    df : polars.LazyFrame | polars.DataFrame
        polars LazyFrame or DataFrame to aggregate features on. If a DataFrame is passed, it will be
        converted to a LazyFrame.
    num_cols : list
        List of numeric columns to aggregate
    groupby_cols : list
        List of categorical columns to group by
    func : function
        Function to use for aggregation
    func_name : str
        Name of the function to use for aggregation
    agg_func : Callable | List[Callable], optional
        Function or list of functions to use for aggregating the created columns that takes the created
        columns in order and returns a column that combines or aggregates the created columns into
        either a single column or multiple columns based on the length of the list of functions. If None,
        this final aggregation step will be skipped.
    agg_func_name : str | List[str], optional
        Name or list of names of the function(s) to use for aggregating the created columns. If None,
        the name of the function(s) will be used, or if no agg_func is provided, this parameter is ignored.
    include_relativities : bool, optional
        Whether or not to include relativities in the aggregation. If True, the relativities will be
        calculated relating the observation value to the aggregated value. If False, the relativities
        will not be calculated. Default is False.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with aggregated features
    """
    # Handle input - if DataFrame is passed, convert to LazyFrame
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    # Handle missing values
    for c in num_cols:
        df = df.with_columns(pl.col(c).cast(pl.Float64).fill_null(0).fill_nan(0))
    for c in groupby_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).fill_null("none"))

    # Aggregate features
    aggs = []
    for gb_col in groupby_cols:
        for num_col in num_cols:
            temp_agg = (
                df.select(
                    [
                        pl.col(gb_col).fill_null("none"),
                        pl.col(num_col).drop_nans().drop_nulls(),
                    ]
                )
                .groupby(gb_col)
                .agg(
                    func(pl.col(num_col))
                    .alias(f"{func_name}_{[gb_col]}({[num_col]})".replace("'", ""))
                    .cast(pl.Float64)
                    .fill_null(0)
                )
            )

            aggs.append(temp_agg)
    agg_df = df.select([pl.col(c) for c in groupby_cols])

    # Join aggregated features back to original DataFrame
    agg_cols = []
    for i, agg in enumerate(aggs):
        agg_cols.append(agg.columns[1])
        if i == 0:
            out_df = df.join(
                agg.select(agg.columns).with_columns(
                    pl.col(agg.columns[1]).fill_nan(0)
                ),
                on=agg.columns[0],
                how="left",
            )
        else:
            out_df = out_df.join(
                agg.select(agg.columns).with_columns(
                    pl.col(agg.columns[1]).fill_nan(0)
                ),
                on=agg.columns[0],
                how="left",
            )

    # Aggregate the created columns
    if agg_func is None:
        pass
    else:
        if isinstance(agg_func, list):
            for i, f in enumerate(agg_func):
                out_df = out_df.with_columns(
                    f(pl.col(agg_cols[i]), pl.col(agg_cols[i + 1]))
                    .alias(f"{agg_func_name[i]}_{[agg_cols[i]]}({[agg_cols[i+1]]})")
                    .cast(pl.Float64)
                    .fill_null(0)
                )
        else:
            out_df = out_df.with_columns(
                agg_func(pl.col(agg_cols))
                .alias(f"{agg_func_name}_{agg_cols}")
                .cast(pl.Float64)
                .fill_null(0)
            )

    # Calculate relativities
    if include_relativities:
        for num, cat in product(num_cols, groupby_cols):
            colname = f"{func_name}_{[cat]}({[num]})".replace("'", "")
            out_df = out_df.with_columns(
                [
                    pl.col(num).alias(f"[{num}] : [{cat}].{func_name}").fill_nan(0)
                    / pl.col(colname)
                ]
            )

    return out_df


def aggregate_ratios(
    df: Union[pl.LazyFrame, pl.DataFrame],
    num_cols: List[str],
    groupby_cols: List[str],
    include_ratios: bool = True,
) -> pl.LazyFrame:
    """
    Aggregate ratios of numeric features by categorical features, and join the aggregated features back to
    the original DataFrame.

    Parameters
    ----------
    df : polars.LazyFrame | polars.DataFrame
        polars LazyFrame or DataFrame to aggregate features on. If a DataFrame is passed, it will be
        converted to a LazyFrame.
    num_cols : list
        List of numeric columns to aggregate
    groupby_cols : list
        List of categorical columns to group by
    include_ratios : bool, optional
        Whether or not to include ratios in the aggregation. Default is True.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with aggregated features
    """
    # Handle input - if DataFrame is passed, convert to LazyFrame
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    # Select columns
    df = df.select([pl.col(n) for n in num_cols] + [pl.col(g) for g in groupby_cols])

    # Handle missing values
    for c in num_cols:
        df = df.with_columns(pl.col(c).cast(pl.Float64).fill_null(0).fill_nan(0))
    for c in groupby_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).fill_null("none"))

    # Aggregate features
    aggs = []
    for gb_col in groupby_cols:
        temp_agg = (
            df.select(
                [
                    pl.col(gb_col).fill_null("none"),
                    pl.col(num_cols[0]).drop_nans().drop_nulls(),
                    pl.col(num_cols[1]).drop_nans().drop_nulls(),
                ]
            )
            .groupby(gb_col)
            .agg(
                pl.col(num_cols[0]).sum()
                / pl.col(num_cols[1])
                .sum()
                .alias(f"avg_ratio_{gb_col}")
                .cast(pl.Float64)
                .fill_null(0)
            )
        )

        aggs.append(temp_agg)

    # Join aggregated features back to original DataFrame
    for i, agg in enumerate(aggs):
        if i == 0:
            out_df = df.join(
                agg.select(agg.columns).with_columns(
                    pl.col(agg.columns[1]).fill_nan(0)
                ),
                on=agg.columns[0],
                how="left",
            )
        else:
            out_df = out_df.join(
                agg.select(agg.columns).with_columns(
                    pl.col(agg.columns[1]).fill_nan(0)
                ),
                on=agg.columns[0],
                how="left",
            )

    return out_df


# TODO: add a default dict for the non_cat_cols and numeric_cat_cols arguments
# TODO: (potentially = [])??
# DEFAULTS =

transformations = namedtuple(
    "transformations",
    [
        "log",
        "inv",
        "sqrt",
        "cbrt",
        "polynomial2",
        "polynomial3",
        "polynomial4",
        "polynomial5",
        "fourier1",
        "fourier2",
        "fourier3",
    ],
)


class FeatureEngineering:
    def __init__(
        self,
        df: Union[pl.LazyFrame, pd.DataFrame, pl.DataFrame],
        # list of columns that uniquely identify a row
        key_cols: List[str],
        # column to be predicted
        target_col: str = "evolve_hit_count",
        # column to be used as the denominator, if any
        denom_col: str = None,
        # list of columns that are not to be considered categories for the
        # purpose of the feature engineering done in this class, regardless
        # of their dtype (i.e. categorical columns that are to be treated as
        # non-feature string columns)
        non_cat_cols: List[str] = None,
        # list of columns that are to be considered categories for the purpose
        # of the feature engineering done in this class, regardless of their
        # dtype (i.e. numeric columns that are to be treated as categories)
        numeric_cat_cols: List[str] = None,
        # data path
        output_data_path: str = "./data/features/",
    ):
        self.df = df
        self.key_cols = key_cols
        self.target_col = target_col
        self.denom_col = denom_col
        self.non_cat_cols = non_cat_cols
        self.numeric_cat_cols = numeric_cat_cols
        self.output_data_path = output_data_path

        # ensure the output folder exists
        if not os.path.exists(self.output_data_path):
            os.makedirs(self.output_data_path)

        # initialize named tuple of transformations
        self.transformations = transformations(
            log=False,
            inv=False,
            sqrt=False,
            cbrt=False,
            polynomial2=False,
            polynomial3=False,
            polynomial4=False,
            polynomial5=False,
            fourier1=False,
            fourier2=False,
            fourier3=False,
        )

        # convert to polars lazyframe if not already
        if isinstance(self.df, pd.DataFrame):
            self.df = pl.from_pandas(
                self.df.convert_dtypes(dtype_backend="pyarrow")
            ).lazy()
        elif isinstance(self.df, pl.DataFrame):
            self.df = self.df.lazy()

        # add a "join" column
        # self.df = self.df.with_columns(pl.lit(1).alias("join"))\
        #                  .with_columns(pl.col('join').cumsum())

        # get a separate key lazyframe
        self.key_df = self.df.select(self.key_cols)

        # get the categorical columns
        self.cat_cols = [
            col
            for col in self.df.columns
            if (self.df.collect()[col].dtype == ("object"))
            or (self.df.collect()[col].dtype == ("string"))
            or (self.df.collect()[col].dtype == ("category"))
        ]

        # add the columns from _feature_engineering_params.NUMERIC_CATEGORICAL_COLUMNS()
        # to the categorical columns (as long as they are not already in the list)
        self.cat_cols = self.cat_cols + [
            col for col in NUMERIC_CATEGORICAL_COLUMNS() if col not in self.cat_cols
        ]

        # remove any categorical columns that are in the NON_CATEGORICAL_COLUMNS list
        self.cat_cols = [
            col for col in self.cat_cols if col not in NON_CATEGORICAL_COLUMNS()
        ]

        # get the numeric columns excluding the key columns, the target column,
        # the denominator column, and the `numeric_cat_cols` columns
        self.num_cols = (
            [
                col
                for col in self.df.columns
                if (self.df.collect()[col].dtype == "int")
                or (self.df.collect()[col].dtype == "float")
            ]
            if self.numeric_cat_cols is None
            else [
                col
                for col in self.df.columns
                if (self.df.collect()[col].dtype == "int")
                or (self.df.collect()[col].dtype == "float")
                and (col not in self.numeric_cat_cols)
            ]
        )

        # get the numeric columns that are specifically excluded in the
        # _feature_engineering_params.py file
        self.num_cols = [
            col for col in self.num_cols if col not in NON_NUMERIC_COLUMNS()
        ]

    def train_test_split(self):
        """
        Adds a column to self.df that indicates whether the observation is in the
        train set, validation set, or
        """

    def Reset(self):
        # initialize named tuple of transformations
        self.transformations = transformations(
            log=False,
            inv=False,
            sqrt=False,
            cbrt=False,
            polynomial2=False,
            polynomial3=False,
            polynomial4=False,
            polynomial5=False,
            fourier1=False,
            fourier2=False,
            fourier3=False,
        )

    def _check_columns(self, df: pl.LazyFrame, search_str: Union[str, list] = "log"):
        # pdb.set_trace()
        # drop all non-log columns before hstacking
        for c in tqdm(df.columns, desc=f"Dropping {search_str} columns"):
            should_keep = False
            if isinstance(search_str, str):
                if c.find(search_str) != -1:
                    should_keep = True
            elif isinstance(search_str, list):
                for s in search_str:
                    if c.find(s) != -1:
                        should_keep = True

            if not should_keep:
                df = df.drop(c)

            else:
                if c.lower().find("missing") != -1:
                    df = df.drop(c)
                elif c.lower().find("naics") != -1:
                    df = df.drop(c)
                elif c.lower().find("class_code") != -1:
                    df = df.drop(c)
                elif c.lower().find("date") != -1:
                    df = df.drop(c)
                elif c.lower().find("dt") != -1:
                    df = df.drop(c)
                elif c.lower().find("ind") != -1:
                    df = df.drop(c)
                elif c.lower().find("id") != -1:
                    df = df.drop(c)
                elif c.lower().find("terr") != -1:
                    df = df.drop(c)
                elif c.lower().find("log(is_") != -1:
                    df = df.drop(c)
                elif c.lower().find("zip") != -1:
                    df = df.drop(c)
                elif c.lower().find("policy_n") != -1:
                    df = df.drop(c)
                elif c.lower().find("policy_m") != -1:
                    df = df.drop(c)
                elif c.lower().find("_bin)") != -1:
                    df = df.drop(c)

        return df

    def AddLog(self, return_: bool = False):
        """
        Adds log features to the dataframe. If the dataframe already has log
        features, this method will do nothing.

        Parameters
        ----------
        return_ : bool, optional
            Whether to return the lazyframe after the transformation is complete.
            The default is False, in which case the lazyframe will only be updated
            in place.

        Returns
        -------
        self.df : pl.LazyFrame
            The lazyframe with the log features added. Will always be updated in
            place, but will also be returned if `return_` is True.
        """
        if self.transformations.log:
            pass
        else:
            # do not transform categories: only transform numeric columns
            cat_cols = [
                field
                for field, dtype in self.df.schema.items()
                if dtype == pl.Categorical or dtype == pl.Object or dtype == pl.Utf8
            ]

            # fill nulls with -1 and add 2 to all values (so no negative values get
            # log transformed)
            col_list = []
            for col in self.df.columns:
                if col not in cat_cols:
                    col_list.append((pl.col(col).fill_null(-1) + 2).alias(col))
                else:
                    pass

            log_df = self.df.select(col_list)

            # add log features & rename columns
            log_df = log_df.with_columns(
                [pl.col(col).log().alias(f"log({col})") for col in log_df.columns]
            )

            # drop all non-log columns before hstacking
            self.log_df = self._check_columns(df=log_df, search_str="log")

            # # hstack the log_df with the original df
            # self.df = self.df.collect().hstack(log_df.collect()).lazy()

            # save the log_df to disk
            self.log_df.collect().write_parquet(
                f"{self.output_data_path}log_df.parquet"
            )

            # update the transformations named tuple so this transformation is not
            # repeated over and over
            self.transformations = self.transformations._replace(log=True)

        # return the lazyframe if requested
        if return_:
            return self.log_df

    def AddInverse(self, return_: bool = False):
        """
        Adds inverse features to the dataframe. If the dataframe already has
        added inverse features, this method will do nothing.

        Parameters
        ----------
        return_ : bool, optional
            Whether to return the lazyframe after the transformation is complete.
            The default is False, in which case the lazyframe will only be updated
            in place.

        Returns
        -------
        self.df : pl.LazyFrame
            The lazyframe with the log features added. Will always be updated in
            place, but will also be returned if `return_` is True.
        """
        if self.transformations.inv:
            pass
        else:
            # do not transform categories: only transform numeric columns
            cat_cols = [
                field
                for field, dtype in self.df.schema.items()
                if dtype == pl.Categorical or dtype == pl.Object or dtype == pl.Utf8
            ]

            # fill nulls with -1 and add 2 to all values (so no division by zero)
            col_list = []
            for col in self.df.columns:
                if col not in cat_cols:
                    col_list.append((pl.col(col).fill_null(-1) + 2).alias(col))
                else:
                    pass

            inv_df = self.df.select(col_list)

            # add inverse features & rename columns
            inv_df = inv_df.with_columns(
                [pl.col(col).pow(-1).alias(f"1/{col}") for col in inv_df.columns]
            )

            # drop all non-inverse columns before hstacking
            self.inv_df = self._check_columns(df=inv_df, search_str="1/")

            # # hstack the inv_df with the original df
            # self.df = self.df.collect().hstack(inv_df.collect()).lazy()

            # save the inv_df to disk
            self.inv_df.collect().write_parquet(
                f"{self.output_data_path}inv_df.parquet"
            )

            # update the transformations named tuple so this transformation is not
            # repeated over and over
            self.transformations = self.transformations._replace(inv=True)

        # return the lazyframe if requested
        if return_:
            return self.inv_df

    def AddSquareRoot(self, return_: bool = False):
        if self.transformations.sqrt:
            pass
        else:
            # do not transform categories: only transform numeric columns
            cat_cols = [
                field
                for field, dtype in self.df.schema.items()
                if dtype == pl.Categorical or dtype == pl.Object or dtype == pl.Utf8
            ]

            # fill nulls with -1 and add 2 to all values (so no division by zero)
            col_list = []
            for col in self.df.columns:
                if col not in cat_cols:
                    col_list.append((pl.col(col).fill_null(-1) + 2).alias(col))
                else:
                    pass

            sqrt_df = self.df.select(col_list)

            # add square root features
            # sqrt_df = sqrt_df.select([pl.col(col)\
            #                             .alias(f"sqrt({col})")\
            #                             .fill_null(0) + 1 \
            #                             for col in self.num_cols])
            sqrt_df = sqrt_df.with_columns(
                [pl.col(col).sqrt().alias(f"sqrt({col})") for col in sqrt_df.columns]
            )

            # drop all non-square root columns before hstacking
            self.sqrt_df = self._check_columns(df=sqrt_df, search_str="sqrt")

            # self.df = self.df.collect().hstack(sqrt_df.collect()).lazy()

            # save the sqrt_df to disk
            self.sqrt_df.collect().write_parquet(
                f"{self.output_data_path}sqrt_df.parquet"
            )

            self.transformations = self.transformations._replace(sqrt=True)

        if return_:
            return self.sqrt_df

    def AddCubeRoot(self, return_: bool = False):
        if self.transformations.cbrt:
            pass
        else:
            # do not transform categories: only transform numeric columns
            cat_cols = [
                field
                for field, dtype in self.df.schema.items()
                if dtype == pl.Categorical or dtype == pl.Object or dtype == pl.Utf8
            ]

            # fill nulls with -1 and add 2 to all values (so no division by zero)
            col_list = []
            for col in self.df.columns:
                if col not in cat_cols:
                    col_list.append((pl.col(col).fill_null(-1) + 2).alias(col))
                else:
                    pass

            cbrt_df = self.df.select(col_list)

            # add cube root features
            # cbrt_df = cbrt_df.select([pl.col(col)\

            #                             .fill_null(0) + 1 \
            #                             for col in self.num_cols])
            cbrt_df = cbrt_df.with_columns(
                [pl.col(col).cbrt().alias(f"cbrt({col})") for col in cbrt_df.columns]
            )

            # drop all non-cube root columns before hstacking
            self.cbrt_df = self._check_columns(df=cbrt_df, search_str="cbrt")

            # save the cbrt_df to disk
            self.cbrt_df.collect().write_parquet(
                f"{self.output_data_path}cbrt_df.parquet"
            )

            # self.df = self.df.collect().hstack(cbrt_df.collect()).lazy()
            self.transformations = self.transformations._replace(cbrt=True)

        if return_:
            return self.cbrt_df

    def AddPolynomial2(self, return_: bool = False):
        from itertools import product

        if self.transformations.polynomial2:
            pass
        else:
            # do not transform categories: only transform numeric columns
            cat_cols = [
                field
                for field, dtype in self.df.schema.items()
                if dtype == pl.Categorical
                or dtype == pl.Object
                or dtype == pl.Utf8
                or dtype == pl.Date
            ]

            # fill nulls with -1 and add 2 to all values (so no division by zero)
            col_list = []
            for i, col in enumerate(self.df.columns):
                if (
                    (col not in cat_cols)
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("dat")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("_dt")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("_id")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("_ind")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("missing")
                    ).tolist()
                    and (self.df.schema[col] != pl.Date)
                    and (self.df.schema[col] != pl.Datetime)
                    and (self.df.schema[col] != pl.Boolean)
                    and (self.df.schema[col] != pl.Categorical)
                    and (self.df.schema[col] != pl.Object)
                    and (self.df.schema[col] != pl.Utf8)
                ):
                    col_list.append(
                        (pl.col(col).cast(pl.Float32).fill_null(-1) + 2).alias(col)
                    )
                else:
                    pass

            poly2_select = []
            cols = [
                col for col in self.df.select(col_list).columns if col not in cat_cols
            ]
            for col1, col2 in tqdm(
                product(cols, cols),
                total=len(cols) ** 2,
                desc="Building polynomial2 features",
            ):
                if col1 != col2:
                    poly2_select.append(
                        (
                            pl.col(col1).fill_nan(0).fill_null(0)
                            * pl.col(col2).fill_nan(0).fill_null(0)
                        ).alias(f"[{col1}]*[{col2}]")
                    )
                else:
                    if col1 in [col for col in self.num_cols if col not in cat_cols]:
                        poly2_select.append(
                            (pl.col(col1).fill_nan(0).fill_null(0).pow(2)).alias(
                                f"[{col1}]^2"
                            )
                        )
                    else:
                        pass

            self.poly2_df = self.df.select(poly2_select)

            # drop all non-poly2 columns before hstacking
            # print("Dropping non-poly2 columns")
            # self.poly2_df = self._check_columns(df=poly2_df, search_str=["^2", "*"])

            # save the poly2_df to disk
            self.poly2_df.collect().write_parquet(
                f"{self.output_data_path}poly2_df.parquet"
            )

            # print('Combining poly2_df with self.df')
            # self.df = self.df.collect().hstack(poly2_df.collect()).lazy()

            self.transformations = self.transformations._replace(polynomial2=True)

        if return_:
            return self.poly2_df

    def AddPolynomial3(self, return_: bool = False):
        self.transformations = self.transformations._replace(polynomial3=False)
        raise NotImplementedError

    def AddPolynomial4(self, return_: bool = False):
        self.transformations = self.transformations._replace(polynomial4=False)
        raise NotImplementedError

    def AddPolynomial5(self, return_: bool = False):
        self.transformations = self.transformations._replace(polynomial5=False)
        raise NotImplementedError

    def AddFourier1(self, return_: bool = False):
        if self.transformations.fourier1:
            pass
        else:
            # do not transform categories: only transform numeric columns
            cat_cols = [
                field
                for field, dtype in self.df.schema.items()
                if dtype == pl.Categorical or dtype == pl.Object or dtype == pl.Utf8
            ]

            # do not need to fill nulls
            col_list = []
            for i, col in enumerate(self.df.columns):
                if (
                    (col not in cat_cols)
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("dat")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("_dt")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("_id")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("_ind")
                    ).tolist()
                    and (
                        ~pd.Series(self.df.columns).str.lower().str.contains("missing")
                    ).tolist()
                    and (self.df.schema[col] != pl.Date)
                    and (self.df.schema[col] != pl.Datetime)
                    and (self.df.schema[col] != pl.Boolean)
                    and (self.df.schema[col] != pl.Categorical)
                    and (self.df.schema[col] != pl.Object)
                    and (self.df.schema[col] != pl.Utf8)
                ):
                    col_list.append(pl.col(col).cast(pl.Float32).alias(col))
                else:
                    pass

            fourier1_df = self.df.select(col_list)

            fourier1_select = []
            for col in fourier1_df.columns:
                fourier1_select.append(
                    (pl.col(col).fill_nan(0).fill_null(0)).sin().alias(f"sin([{col}])")
                )
                fourier1_select.append(
                    (pl.col(col).fill_nan(0).fill_null(0)).cos().alias(f"cos([{col}])")
                )
            fourier1_df = self.df.select(fourier1_select)

            # drop all non-inverse columns before hstacking
            self.fourier1_df = self._check_columns(
                df=fourier1_df, search_str=["sin", "cos"]
            )

            # self.df = self.df.collect().hstack(fourier1_df.collect()).lazy()

            # save the fourier1_df to disk
            self.fourier1_df.collect().write_parquet(
                f"{self.output_data_path}fourier1_df.parquet"
            )

            self.transformations = self.transformations._replace(fourier1=True)

        if return_:
            return self.fourier1_df

    def AddFourier2(self, return_: bool = False):
        if self.transformations.fourier2:
            pass
        else:
            fourier2_select = []
            for col1 in self.num_cols:
                for col2 in self.num_cols:
                    if col1 != col2:
                        fourier2_select.append(
                            (
                                pl.col(col1).fill_nan(0).fill_null(0)
                                * pl.col(col2).fill_nan(0).fill_null(0)
                            )
                            .sin()
                            .alias(f"sin([{col1}] * [{col2}])")
                        )
                        fourier2_select.append(
                            (
                                pl.col(col1).fill_nan(0).fill_null(0)
                                * pl.col(col2).fill_nan(0).fill_null(0)
                            )
                            .cos()
                            .alias(f"cos([{col1}] * [{col2}])")
                        )
                    else:
                        if col1 in self.num_cols:
                            # cos^2 term
                            fourier2_select.append(
                                (
                                    pl.col(col1).fill_nan(0).fill_null(0).pow(2).cos()
                                ).alias(f"cos([{col1}]^2)")
                            )

                            # sin^2 term
                            fourier2_select.append(
                                (
                                    pl.col(col1).fill_nan(0).fill_null(0).pow(2).sin()
                                ).alias(f"sin([{col1}]^2)")
                            )

                            # sin * cos term
                            fourier2_select.append(
                                (pl.col(col1).fill_nan(0).fill_null(0).sin())
                                * (pl.col(col1).fill_nan(0).fill_null(0).cos()).alias(
                                    f"sin([{col1}])*cos([{col2}])"
                                )
                            )
                        else:
                            pass

            fourier2_df = self.df.select(fourier2_select)

            # drop all non-fourier2 columns before hstacking
            self.fourier2_df = self._check_columns(
                df=fourier2_df, search_str=["sin", "cos"]
            )

            # self.df = self.df.collect().hstack(fourier2_df.collect()).lazy()

            # save the fourier2_df to disk
            self.fourier2_df.collect().write_parquet(
                f"{self.output_data_path}fourier2_df.parquet"
            )

            self.transformations = self.transformations._replace(fourier2=True)

        if return_:
            return self.df

    def categorical_encoding(
        self,
        func: Callable,
        func_name: str,
        df: Optional[pl.LazyFrame] = None,
        agg_func: Optional[Union[Callable, List[Callable]]] = None,
        agg_func_name: Optional[Union[str, List[str]]] = None,
        include_relativities: bool = True,
    ):
        ncol = []
        ccol = []
        if df is None:
            df0 = self.df
        else:
            df0 = df

        for col in df0.columns:
            if (
                (col not in self.cat_cols)
                and (~pd.Series(df0.columns).str.lower().str.contains("dat")).tolist()
                and (~pd.Series(df0.columns).str.lower().str.contains("_dt")).tolist()
                and (~pd.Series(df0.columns).str.lower().str.contains("_id")).tolist()
                and (~pd.Series(df0.columns).str.lower().str.contains("_ind")).tolist()
                and (
                    ~pd.Series(df0.columns).str.lower().str.contains("missing")
                ).tolist()
                and (df0.schema[col] != pl.Date)
                and (df0.schema[col] != pl.Datetime)
                and (df0.schema[col] != pl.Boolean)
                and (df0.schema[col] != pl.Categorical)
                and (df0.schema[col] != pl.Object)
                and (df0.schema[col] != pl.Utf8)
            ):
                ncol.append(col)
            elif (
                (~pd.Series(df0.columns).str.lower().str.contains("dat")).tolist()
                and (~pd.Series(df0.columns).str.lower().str.contains("_dt")).tolist()
                and (~pd.Series(df0.columns).str.lower().str.contains("_id")).tolist()
            ) and (
                (df0.schema[col] == pl.Categorical)
                or (df0.schema[col] == pl.Object)
                or (df0.schema[col] == pl.Boolean)
                or (df0.schema[col] == pl.Utf8)
            ):
                ccol.append(col)

        out = aggregate_features(
            df=df0,
            num_cols=ncol,
            groupby_cols=self.cat_cols,
            func=func,
            func_name=func_name,
            agg_func=agg_func,
            agg_func_name=agg_func_name,
            include_relativities=include_relativities,
        )

        return out

    def SumCategoricalEncoding(self, return_: bool = False):
        self.sum_cat_df = self.categorical_encoding(
            func=lambda x: x.sum(), func_name="sum", include_relativities=True
        )

        self.sum_cat_df.collect().write_parquet(
            f"{self.output_data_path}sum_cat_df.parquet"
        )

        if return_:
            return self.sum_cat_df

    def MeanCategoricalEncoding(self, return_: bool = False):
        self.mean_cat_df = self.categorical_encoding(
            func=lambda x: x.mean(), func_name="mean", include_relativities=True
        )

        self.mean_cat_df.collect().write_parquet(
            f"{self.output_data_path}mean_cat_df.parquet"
        )

        if return_:
            return self.mean_cat_df

    def MedianCategoricalEncoding(self, return_: bool = False):
        self.median_cat_df = self.categorical_encoding(
            func=lambda x: x.median(), func_name="median", include_relativities=True
        )

        self.median_cat_df.collect().write_parquet(
            f"{self.output_data_path}median_cat_df.parquet"
        )

        if return_:
            return self.median_cat_df

    def MinCategoricalEncoding(self, return_: bool = False):
        self.min_cat_df = self.categorical_encoding(
            func=lambda x: x.min(), func_name="min", include_relativities=True
        )

        self.min_cat_df.collect().write_parquet(
            f"{self.output_data_path}min_cat_df.parquet"
        )

        if return_:
            return self.min_cat_df

    def MaxCategoricalEncoding(self, return_: bool = False):
        self.max_cat_df = self.categorical_encoding(
            func=lambda x: x.max(), func_name="max", include_relativities=True
        )

        self.max_cat_df.collect().write_parquet(
            f"{self.output_data_path}max_cat_df.parquet"
        )

        if return_:
            return self.max_cat_df

    def StdCategoricalEncoding(self, return_: bool = False):
        self.std_cat_df = self.categorical_encoding(
            func=lambda x: x.std(), func_name="std", include_relativities=True
        )

        self.std_cat_df.collect().write_parquet(
            f"{self.output_data_path}std_cat_df.parquet"
        )

        if return_:
            return self.std_cat_df

    def VarCategoricalEncoding(self, return_: bool = False):
        self.var_cat_df = self.categorical_encoding(
            func=lambda x: x.var(), func_name="var", include_relativities=True
        )

        self.var_cat_df.collect().write_parquet(
            f"{self.output_data_path}var_cat_df.parquet"
        )

        if return_:
            return self.var_cat_df

    def SkewCategoricalEncoding(self, return_: bool = False):
        self.skew_cat_df = self.categorical_encoding(
            func=lambda x: x.skew(), func_name="skew", include_relativities=True
        )

        self.skew_cat_df.collect().write_parquet(
            f"{self.output_data_path}skew_cat_df.parquet"
        )

        if return_:
            return self.skew_cat_df

    def KurtosisCategoricalEncoding(self, return_: bool = False):
        self.kurtosis_cat_df = self.categorical_encoding(
            func=lambda x: x.kurtosis(), func_name="kurtosis", include_relativities=True
        )

        self.kurtosis_cat_df.collect().write_parquet(
            f"{self.output_data_path}kurtosis_cat_df.parquet"
        )

        if return_:
            return self.kurtosis_cat_df

    def IQRCategoricalEncoding(self, return_: bool = False):
        self.iqr_cat_df = self.categorical_encoding(
            func=lambda x: x.quantile(0.75) - x.quantile(0.25),
            func_name="iqr",
            include_relativities=True,
        )

        self.iqr_cat_df.collect().write_parquet(
            f"{self.output_data_path}iqr_cat_df.parquet"
        )

        if return_:
            return self.iqr_cat_df

    def HarmonicMeanCategoricalEncoding(self, return_: bool = False):
        self.harmonic_mean_cat_df = self.categorical_encoding(
            func=lambda x: 1 / x.pow(-1).mean(),
            func_name="harmonic_mean",
            include_relativities=True,
        )

        self.harmonic_mean_cat_df.collect().write_parquet(
            f"{self.output_data_path}harmonic_mean_cat_df.parquet"
        )

        if return_:
            return self.harmonic_mean_cat_df

    def AddCategoricalEncoding(self):
        encodings = [
            self.SumCategoricalEncoding,
            self.MeanCategoricalEncoding,
            self.MedianCategoricalEncoding,
            self.MinCategoricalEncoding,
            self.MaxCategoricalEncoding,
            self.StdCategoricalEncoding,
            self.VarCategoricalEncoding,
            self.SkewCategoricalEncoding,
            self.KurtosisCategoricalEncoding,
            self.IQRCategoricalEncoding,
            self.HarmonicMeanCategoricalEncoding,
        ]

        for encoding in tqdm(encodings, desc="Adding categorical encodings"):
            encoding()
