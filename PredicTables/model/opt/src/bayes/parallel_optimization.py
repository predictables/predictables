import skopt
from typing import Callable, Any


def parallel_optimization(
    objective_func: Callable[..., Any],
    space: skopt.Space,
    n_calls: int,
    batch_size: int = 4,
    backend: str = "multiprocessing",
) -> skopt.Optimizer:
    """
    Run Bayesian optimization in parallel. The objective function is evaluated
    in parallel using either a process pool or a thread pool.

    Parameters
    ----------

    objective_func : callable
        Objective function to be optimized.
    space : skopt.Space
        Search space.
    n_calls : int
        Number of calls to `objective_func`.
    batch_size : int, optional
        Number of points to evaluate in parallel. Defaults to 4.
    backend : str, optional
        Parallelization backend. Either "multiprocessing" or "threading".
        Defaults to "multiprocessing". Note that "threading" is only useful
        when `objective_func` is not CPU-bound. This is rarely the case in
        hyperparameter optimization.

    Returns
    -------
    optimizer : skopt.Optimizer
        Optimizer instance.

    Examples
    --------
    >>> import skopt
    >>> import numpy as np
    >>> from sklearn.datasets import load_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
    >>> X, y = load_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)

    >>> def objective_func(params):
    ...     C, gamma = params
    ...     model = Pipeline([
    ...         ("scaler", StandardScaler()),
    ...         ("pca", PCA(n_components=2)),
    ...         ("svm", SVC(C=C, gamma=gamma, random_state=0, probability=True)),
    ...     ])
    ...     score = cross_val_score(
    ...         model, X_train, y_train, cv=3, scoring=make_scorer(roc_auc_score)
    ...     ).mean()
    ...     return 1 - score

    >>> space = [
    ...     skopt.space.Real(1e-6, 1e+6, prior="log-uniform", name="C"),
    ...     skopt.space.Real(1e-6, 1e+1, prior="log-uniform", name="gamma"),
    ... ]

    >>> optimizer = parallel_optimization(
    ...     objective_func, space, n_calls=20, batch_size=4, backend="multiprocessing"
    ... )
    >>> best_params = optimizer.x
    >>> best_score = optimizer.fun
    >>> best_params, best_score
    ([0.0001, 0.0001], 0.012...)

    See Also
    --------
    skopt.Optimizer : Optimizer class.
    skopt.space : Search space.

    Notes
    -----
    This function is a simplified version of the `parallel_optimization`
    function from the `skopt` library. It is meant to be used as a drop-in
    replacement for `skopt.gp_minimize` and `skopt.forest_minimize` when
    running on a cluster. The `objective_func` is evaluated in parallel using
    either a process pool or a thread pool. The `batch_size` parameter
    controls the number of points to evaluate in parallel. The `backend`
    parameter controls the parallelization backend. Either "multiprocessing"
    or "threading". Note that "threading" is only useful when `objective_func`
    is not CPU-bound. This is rarely the case in hyperparameter optimization.

    References
    ----------
    https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
    https://scikit-optimize.github.io/stable/modules/generated/skopt.forest_minimize.html
    https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html
    https://scikit-optimize.github.io/stable/modules/generated/skopt.space.html
    """
    if backend == "multiprocessing":
        return _parallel_optimization_process_pool(
            objective_func, space, n_calls, batch_size
        )
    elif backend == "threading":
        return _parallel_optimization_thread_pool(
            objective_func, space, n_calls, batch_size
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Please choose between 'multiprocessing' and 'threading'."
        )


def _parallel_optimization_process_pool(
    objective_func: Callable[..., Any],
    space: skopt.Space,
    n_calls: int,
    batch_size: int,
) -> skopt.Optimizer:
    """
    Internal function to run Bayesian optimization in parallel using a process
    pool. Mainly used by the `parallel_optimization` function. Not meant to be
    used directly.
    """
    from concurrent.futures import ProcessPoolExecutor

    optimizer = skopt.Optimizer(space)
    for _ in range(n_calls // batch_size):
        points = [optimizer.ask() for _ in range(batch_size)]

        with ProcessPoolExecutor() as executor:
            evaluations = list(executor.map(objective_func, points))

        for point, evaluation in zip(points, evaluations):
            optimizer.tell(point, evaluation)

    return optimizer


def _parallel_optimization_thread_pool(
    objective_func: Callable[..., Any],
    space: skopt.Space,
    n_calls: int,
    batch_size: int,
) -> skopt.Optimizer:
    """
    Internal function to run Bayesian optimization in parallel using a thread
    pool. Mainly used by the `parallel_optimization` function. Not meant to be
    used directly.
    """
    from concurrent.futures import ThreadPoolExecutor

    optimizer = skopt.Optimizer(space)
    for _ in range(n_calls // batch_size):
        # Ask for a batch of points
        points = [optimizer.ask() for _ in range(batch_size)]

        # Evaluate points in parallel
        with ThreadPoolExecutor() as executor:
            evaluations = list(executor.map(objective_func, points))

        # Update the model with the results
        for point, evaluation in zip(points, evaluations):
            optimizer.tell(point, evaluation)

    return optimizer
