def get_significance_band(p_value: float, statistic: str) -> str:
    """
    Get a significance band based on the p-value of a statistic.

    Parameters
    ----------
    p_value : float
        The p-value of the statistic
    statistic : str
        The name of the statistic

    Returns
    -------
    str
        The significance band & a statement about the significance of the statistic
    """
    if p_value < 0.01:
        significance_statement = (
            f"Extremely likely that the {statistic} is significant"
        )
    elif p_value < 0.05:
        significance_statement = (
            f"Very likely that the {statistic} is significant"
        )
    elif p_value < 0.10:
        significance_statement = (
            f"Somewhat likely that the {statistic} is significant"
        )
    else:
        significance_statement = (
            f"Unlikely that the {statistic} is significant"
        )
    return significance_statement
