# Function to calculate the KL divergence between two distributions
def kl_divergence(observed: list, modeled: list):
    """Calculate the KL divergence between two distributions. This function is a wrapper for scipy.stats.entropy.

    Parameters
    ----------
    observed : list
        List of observed values
    modeled : list
        List of modeled values

    Returns
    -------
    float
        KL divergence between the two distributions
    """
    from scipy.stats import entropy

    # Calculate KL divergence directly from observed and modeled averages
    return entropy(observed, modeled)
