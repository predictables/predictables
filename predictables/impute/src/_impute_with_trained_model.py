import pandas as pd

from predictables.util import to_pd_df


def impute_single_column(
    df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    column: str,
    trained_model,
    learning_rate: float = 0.1,
    only_missing: bool = True,
) -> pd.DataFrame:
    """
    Impute missing values in a single column of a dataframe using a trained CatBoost model.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.
    missing_mask : pd.DataFrame
        The mask indicating which values are missing in the df.
    column : str
        The name of the column to impute.
    trained_model : Any model that implements the predict method.
        A trained model with a predict method. Default is a CatBoost model.
    learning_rate : float, optional
        The learning rate to use for the CatBoost model. The default is 0.1.
    only_missing : bool, optional
        Whether to only impute missing values or all values. The default is True.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with missing values imputed.
    """
    df = to_pd_df(df)
    missing_mask = to_pd_df(missing_mask)
    # Check inputs
    assert isinstance(
        df, pd.DataFrame
    ), f"df must be a pandas DataFrame, not {type(df)}"
    assert isinstance(
        missing_mask, pd.DataFrame
    ), f"missing_mask must be a pandas DataFrame, not {type(missing_mask)}"
    assert (
        df.shape == missing_mask.shape
    ), f"df and missing_mask must have the same shape. df.shape: {df.shape}, missing_mask.shape: {missing_mask.shape}"

    # if only_missing is true, check that the column has missing values.
    # if not, return the df unadjusted
    if only_missing and not missing_mask[column].any():
        return df

    # Get the current value from the column
    current_value = df[column]

    # Get the current value's data type
    dtype = "n" if df[column].dtype in ["int64", "float64"] else "c"

    # Get the full-credibility imputed value for only the missing rows in the column
    full_cred_estimate = df.copy()
    full_cred_estimate.loc[missing_mask[column], column] = (
        trained_model.predict(
            df.loc[missing_mask[column], ~df.columns.isin([column])]
        )
    )

    # Update the imputed missing value with the full_cred_estimate if dtype is "c"
    if dtype == "c":
        df.loc[missing_mask[column], column] = full_cred_estimate.loc[
            missing_mask[column], column
        ]

    # Update the imputed missing value by adding
    # learning_rate * (full_cred_estimate - current_value) if dtype is "n"
    elif dtype == "n":
        df.loc[missing_mask[column], column] = (
            current_value
            + learning_rate
            * (
                full_cred_estimate.loc[missing_mask[column], column]
                - current_value
            )
        )

    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    return df


def impute_with_trained_model(
    df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    trained_models: dict,
    learning_rate: float = 0.1,
) -> pd.DataFrame:
    """
    Impute missing values in a dataframe using trained CatBoost models.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.
    missing_mask : pd.DataFrame
        The mask indicating which values are missing in the df.
    trained_models : dict
        A dictionary of trained CatBoost models whose keys are the column names of the data frame.
    learning_rate : float, optional
        The learning rate to use for the CatBoost models. The default is 0.1.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with missing values imputed.
    """
    df = to_pd_df(df)
    missing_mask = to_pd_df(missing_mask)
    # Check inputs
    assert isinstance(
        df, pd.DataFrame
    ), f"df must be a pandas DataFrame, not {type(df)}"
    assert isinstance(
        missing_mask, pd.DataFrame
    ), f"missing_mask must be a pandas DataFrame, not {type(missing_mask)}"
    assert (
        df.shape == missing_mask.shape
    ), f"df and missing_mask must have the same shape. df.shape: {df.shape}, missing_mask.shape: {missing_mask.shape}"

    # Loop through each column name from the keys of trained_models
    for column in trained_models:
        # Impute the column with the trained model
        df = impute_single_column(
            df, missing_mask, column, trained_models[column][0], learning_rate
        )

    return df
