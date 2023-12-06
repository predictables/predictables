from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb


def dynamic_ucb(optimizer, x, batch_size=4):
    """
    Uses the dynamic upper confidence bound acquisition function to determine
    the next point to evaluate.

    Parameters
    ----------
    optimizer : skopt.Optimizer
        The optimizer object.
    x : array-like
        The point to evaluate.
    batch_size : int
        The number of points to evaluate in parallel. Defaults to 4.

    Returns
    -------
    next_point : array-like
        The next point to evaluate.

    Notes
    -----
    The dynamic upper confidence bound acquisition function is defined as:

    .. math::

            UCB(x) = \mu(x) + \beta \sigma(x)

        where :math:`\mu(x)` is the mean of the surrogate model, and
        :math:`\sigma(x)` is the standard deviation of the surrogate model.
    """

    # Set up the dynamic upper confidence bound acquisition function
    def acq_func(x):
        return -optimizer._gp.predict(x.reshape(1, -1), return_std=True)[1]

    # Ask for the next point to evaluate
    return optimizer.ask(acq_func=acq_func, n_points=batch_size)


def expected_improvement(optimizer, x, batch_size=4):
    """
    Uses the expected improvement acquisition function to determine the next
    point to evaluate. This is the default acquisition function used by
    scikit-optimize.

    Parameters
    ----------
    optimizer : skopt.Optimizer
        The optimizer object.
    x : array-like
        The point to evaluate.
    batch_size : int
        The number of points to evaluate in parallel. Defaults to 4.

    Returns
    -------
    next_point : array-like
        The next point to evaluate.

    Notes
    -----
    The expected improvement acquisition function is defined as:

    .. math::

            EI(x) = \mathbb{E} \left[ \max(0, f(x) - f(x^+)) \right]

        where :math:`f` is the surrogate model, and :math:`x^+` is the best point

    EI(x) tends to favor points that are close to the current best point, but
    with high uncertainty. This encourages exploration near the current best
    point, but also exploitation of points that are likely to be better than
    the current best point.
    """

    # Set up the expected improvement acquisition function
    def acq_func(x):
        return -gaussian_ei(x.reshape(1, -1), optimizer._gp)

    # Ask for the next point to evaluate
    return optimizer.ask(acq_func=acq_func, n_points=batch_size)


def probability_of_improvement(optimizer, x, xi=0.01, batch_size=4):
    """
    Uses the probability of improvement acquisition function to determine the
    next point to evaluate.

    Parameters
    ----------
    optimizer : skopt.Optimizer
        The optimizer object.
    x : array-like
        The point to evaluate.
    xi : float
        Controls the amount of exploration. Defaults to 0.01.
    batch_size : int
        The number of points to evaluate in parallel. Defaults to 4.

    Returns
    -------
    next_point : array-like
        The next point to evaluate.
    """

    # Set up the probability of improvement acquisition function
    def acq_func(x):
        return -gaussian_pi(x.reshape(1, -1), optimizer._gp, xi=xi)

    # Ask for the next point to evaluate
    return optimizer.ask(acq_func=acq_func, n_points=batch_size)


def lower_confidence_bound(optimizer, x, kappa=1.96, batch_size=4):
    """
    Uses the lower confidence bound acquisition function to determine the next
    point to evaluate.

    Parameters
    ----------
    optimizer : skopt.Optimizer
        The optimizer object.
    x : array-like
        The point to evaluate.
    kappa : float
        Controls the amount of exploration. Defaults to 1.96.
    batch_size : int
        The number of points to evaluate in parallel. Defaults to 4.

    Returns
    -------
    next_point : array-like
        The next point to evaluate.
    """

    # Set up the lower confidence bound acquisition function
    def acq_func(x):
        return -gaussian_lcb(x.reshape(1, -1), optimizer._gp, kappa=kappa)

    # Ask for the next point to evaluate
    return optimizer.ask(acq_func=acq_func, n_points=batch_size)
