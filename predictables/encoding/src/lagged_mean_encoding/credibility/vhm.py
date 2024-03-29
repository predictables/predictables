import polars as pl
import polars.selectors as cs

def vhm(lf: pl.LazyFrame, cat_col: str, target_col: str, n_boot: int = 1000, resample_rate: float = 0.75) -> pl.LazyFrame:
    """Estimate the variance of the hypothetical mean for each group.

    Uses bootstrap resampling to estimate the variance of the hypothetical
    mean for each group.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input DataFrame.
    cat_col : str
        The column to group by.
    target_col : str
        The column to calculate the mean of.
    n_boot : int
        The number of bootstrap samples to use.
    resample_rate : float
        The fraction of the DataFrame to sample for each bootstrap sample.

    Returns
    -------
    pl.LazyFrame
        A DataFrame with the variance of the hypothetical mean for each group.
    """
    # Calculate the mean for each group
    means = lf.groupby(cat_col).agg(pl.col(target_col).mean().alias("mean"))

    # Add `n_boot` columns of bootstrap samples
    for i in range(n_boot):
        sample = lf.sample(frac=resample_rate, replace=True)
        sample_means = sample.groupby(cat_col).agg(pl.col(target_col).mean().alias(f"boot_mean_{i}"))
        means = means.join(sample_means, on=cat_col)

    # Use a reduce operation to calculate the variance of the bootstrap samples
    return means.select([cat_col] + [cs.contains("boot_mean")]).with_columns([
        pl.cum_sum_horizontal(means.select(cs.contains("boot_mean")).columns).alias("cum_sum"),
    ]).with_columns([
        (pl.col("cum_sum") / n_boot).alias("cum_mean")
    ]).with_columns([
        pl.reduce(function=lambda acc, x: acc + (x - pl.col("cum_mean")) ** 2, exprs=cs.contains("boot_mean"), init=0).truediv(n_boot - 1).alias("variance")
    ]).drop(cs.contains("boot_mean")).drop("cum_sum").drop("cum_mean")

