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


from .parallel_optimization import parallel_optimization
from .objective_function import objective_with_pruning, _objective_function_no_pruning
from .get_space import get_space
from .get_model_obj import get_model_obj

from PredicTables.util import model_name_map


def get_model_obj(model_name: str, task_type: str = "classification", **kwargs):
    """
    Returns a newly-instantiated model object. Accepts keyword arguments to be
    passed to the model constructor.
    """
    # Check and standardize inputs
    model_name = model_name_map(model_name)

    task_type = task_type.lower().strip()
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            f'Invalid task_type {task_type}. Must be one of "classification" or "regression".'
        )

    if model_name == "catboost":
        if task_type == "classification":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(**kwargs)
        elif task_type == "regression":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(**kwargs)

    elif model_name == "random_forest":
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**kwargs)
        elif task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**kwargs)

    elif model_name == "elastic_net":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(**kwargs)

    elif model_name == "svm":
        if task_type == "classification":
            from sklearn.svm import SVC

            return SVC(**kwargs)
        elif task_type == "regression":
            from sklearn.svm import SVR

            return SVR(**kwargs)

    elif model_name == "lightgbm":
        if task_type == "classification":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(**kwargs)
        elif task_type == "regression":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**kwargs)

    elif model_name == "xgboost":
        if task_type == "classification":
            from xgboost import XGBClassifier

            return XGBClassifier(**kwargs)
        elif task_type == "regression":
            from xgboost import XGBRegressor

            return XGBRegressor(**kwargs)

    else:
        raise ValueError(
            f"Invalid model name {model_name}. Must be one of "
            f"'catboost', 'random_forest', 'elastic_net', 'svm', 'lightgbm', "
            f"or 'xgboost'."
        )


import pandas as pd
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from typing import Union
from PredicTables.util import to_pd_df, to_pd_s


def _objective_function_no_pruning(params):
    """
    This is for testing purposes only. It is not used in the actual
    optimization process. It is a simple objective function that seeks
    to minimize the L1 norm of the parameters.

    Parameters
    ----------
    params : dict
        Hyperparameters for the model.

    Returns
    -------
    float
        The sum of the absolute values of the parameters.
    """
    # Absolute value of the parameters
    abs_params = np.abs(list(params.values()))

    # Return the sum of the absolute values of the parameters
    return np.sum(abs_params)


def objective_with_pruning(
    params: dict,
    model: BaseEstimator,
    X_train: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y_train: Union[pd.Series, pl.Series],
    X_val: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y_val: Union[pd.Series, pl.Series],
    pruning_threshold: float,
    n_checkpoints: int = 10,
) -> float:
    """
    Objective function that includes logic for pruning.

    Parameters
    ----------
    params : dict
        Hyperparameters for the model.
    model : sklearn.base.BaseEstimator
        Machine learning model that supports partial fitting or early stopping,
        typically inheriting from sklearn.base.BaseEstimator.
    X_train, y_train : array-like
        Training features and labels.
    X_val, y_val : array-like
        Validation features and labels. Used for pruning.
    pruning_threshold : float
        Threshold for pruning. If the model's performance on the validation set
        is worse than this threshold, the model is pruned.

    Returns
    -------
    float
        The model's performance on the validation set.
    """
    # Convert to pandas
    X_train = to_pd_df(X_train)
    y_train = to_pd_s(y_train)
    X_val = to_pd_df(X_val)
    y_val = to_pd_s(y_val)

    # Initialize the best score to infinity
    best_evaluation = float("inf")

    # Loop over checkpoints, fitting the model at each checkpoint, evaluating,
    # and making sure the improvement justifies additional training time
    # (i.e., pruning)
    pct_to_train = 1.0 / n_checkpoints
    for checkpoint in range(1, n_checkpoints + 1):
        # Calculate the number of iterations to train for at this checkpoint
        pct_trained = pct_to_train * checkpoint
        n_iterations = int(pct_trained * params["n_iterations"])

        # Update the model with the new number of iterations
        model = partial_fit(X_train, y_train, model, n_iterations=n_iterations)

        # Evaluate the model
        current_evaluation = evaluate_partial_fit(X_val, y_val, model)

        # Pruning condition: if current score is worse than the threshold, prune
        if should_model_be_pruned(
            current_evaluation, best_evaluation, pruning_threshold
        ):
            break

        # If not pruned, update the best score
        best_evaluation = current_evaluation

    # By exiting the loop, we have either pruned the model or trained it to
    # completion. If the model was pruned, we need to retrain it to completion
    # before returning the final evaluation.
    model = complete_fit(X_train, y_train, model, **params)

    # Evaluate the model
    final_evaluation = evaluate_model(X_val, y_val, model)

    # Return the best evaluation if the model was pruned, otherwise return
    # the final evaluation
    final_score = (
        best_evaluation if best_evaluation < final_evaluation else final_evaluation
    )

    return final_score


def partial_fit(X_train, y_train, model):
    # Partially fit the model here
    # Depending on the model, different approaches may be needed
    # such as early stopping, training for a certain number of iterations, etc.
    pass


def evaluate_partial_fit(X_val, y_val, model):
    # Evaluate the model here
    # Evaluate consistently with ultimate goal of optimization process --
    # e.g., if optimizing for accuracy, evaluate using accuracy, not AUC
    pass


def should_model_be_pruned(new_score, current_score, pruning_factor=1, cond="gt"):
    # Define pruning condition here
    # implementing a simple one for now
    if cond == "gt":
        return current_score > (pruning_factor * new_score)
    elif cond == "lt":
        return current_score < (pruning_factor * new_score)
    else:
        raise NotImplementedError(
            f"Condition {cond} not implemented. Currently only 'gt' and 'lt' are supported."
        )


def complete_fit(X_train, y_train, model, **params):
    # Fit the model here
    pass


def evaluate_model(X_val, y_val, model):
    # Evaluate the model here
    # Evaluate consistently with ultimate goal of optimization process --
    # e.g., if optimizing for accuracy, evaluate using accuracy, not AUC
    pass


def adjust_exploration_parameter(iteration, max_iterations, performance_metrics=None):
    # Default Exploration Strategy

    # Initial High Exploration:
    # Start with a higher exploration parameter to ensure a broad search of the space.
    # Gradual Reduction:
    # Slowly reduce the exploration parameter over iterations, shifting towards exploitation.
    # Safety Net:
    # Implement a mechanism where if no improvement is seen for a certain number of
    # iterations, the exploration parameter is temporarily increased to escape potential local optima.
    pass


import skopt
from skopt.space import Real, Integer, Categorical
from typing import List

from PredicTables.util import model_name_map


def get_space(model_name: str, **kwargs) -> List[skopt.Space]:
    """
    Returns a properly-formatted space for hyperparameter optimization.

    Parameters
    ----------
    model_name : str
        Name of the model to optimize. Must be one of "catboost", "random_forest",
        "elastic_net", "svm", "lightgbm", or "xgboost".
    kwargs : dict
        Keyword arguments to be added to the default space, or to use to replace
        default space values.

    Returns
    -------
    space : list of skopt.space objects
        A list of skopt.space objects.

    Examples
    --------
    >>> from PredicTables.model.opt import Bayes
    >>> space = Bayes.get_space("catboost")
    >>> space
    [Integer(low=1, high=15, prior='uniform', transform='identity'),
     Real(low=0.01, high=1.0, prior='uniform', transform='identity'),
     Integer(low=1, high=10, prior='uniform', transform='identity'),
     Categorical(categories=('SymmetricTree', 'Depthwise', 'Lossguide'), prior=None),
     Integer(low=1, high=255, prior='uniform', transform='identity'),
     Real(low=0.0, high=1.0, prior='uniform', transform='identity')]

    >>> space = Bayes.get_space("catboost", learning_rate=Real(0.5, 1.0))
    >>> space
    [Integer(low=1, high=15, prior='uniform', transform='identity'),
     Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
     Integer(low=1, high=10, prior='uniform', transform='identity'),
     Categorical(categories=('SymmetricTree', 'Depthwise', 'Lossguide'), prior=None),
     Integer(low=1, high=255, prior='uniform', transform='identity'),
     Real(low=0.0, high=1.0, prior='uniform', transform='identity')]
    """
    if model_name_map(model_name) == "catboost":
        return get_catboost_space(**kwargs)
    elif model_name_map(model_name) == "random_forest":
        return get_random_forest_space(**kwargs)
    elif model_name_map(model_name) == "elastic_net":
        return get_elastic_net_space(**kwargs)
    elif model_name_map(model_name) == "svm":
        return get_svm_space(**kwargs)
    elif model_name_map(model_name) == "lightgbm":
        return get_lightgbm_space(**kwargs)
    elif model_name_map(model_name) == "xgboost":
        return get_xgboost_space(**kwargs)
    else:
        raise ValueError(
            f"Invalid model name {model_name}. Must be one of "
            f"'catboost', 'random_forest', 'elastic_net', 'svm', 'lightgbm', "
            f"or 'xgboost'."
        )


def get_catboost_space(**kwargs) -> List[skopt.Space]:
    """
    Returns a properly-formatted space for CatBoost hyperparameter optimization.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to be added to the default space, or to use to replace
        default space values.

    Returns
    -------
    space : list of skopt.space objects
        A list of skopt.space objects.

    Examples
    --------
    >>> from PredicTables.model.opt import Bayes
    >>> space = Bayes.get_catboost_space()
    >>> space
    [Integer(low=1, high=15, prior='uniform', transform='identity'),
     Real(low=0.01, high=1.0, prior='uniform', transform='identity'),
     Integer(low=1, high=10, prior='uniform', transform='identity'),
     Categorical(categories=('SymmetricTree', 'Depthwise', 'Lossguide'), prior=None),
     Integer(low=1, high=255, prior='uniform', transform='identity'),
     Real(low=0.0, high=1.0, prior='uniform', transform='identity')]

    >>> space = Bayes.get_catboost_space(learning_rate=Real(0.5, 1.0))
    >>> space
    [Integer(low=1, high=15, prior='uniform', transform='identity'),
     Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
     Integer(low=1, high=10, prior='uniform', transform='identity'),
     Categorical(categories=('SymmetricTree', 'Depthwise', 'Lossguide'), prior=None),
     Integer(low=1, high=255, prior='uniform', transform='identity'),
     Real(low=0.0, high=1.0, prior='uniform', transform='identity')]
    """
    # Define default space
    default_space = {
        "depth": Integer(1, 15),
        "learning_rate": Real(0.01, 1.0),
        "l2_leaf_reg": Integer(1, 10),
        "grow_policy": Categorical(["SymmetricTree", "Depthwise", "Lossguide"]),
        "border_count": Integer(1, 255),
        "bagging_temperature": Real(0.0, 1.0),
    }

    # Update default space with kwargs
    updated_space = {**default_space, **kwargs}

    # Convert to skopt space format
    return [updated_space[key] for key in updated_space]


def get_random_forest_space(**kwargs):
    # Define default space
    default_space = [
        Integer(10, 1000, name="n_estimators"),
        Integer(1, 30, name="max_depth"),
        Real(0.1, 0.9, name="min_samples_split"),
        Integer(1, 60, name="min_samples_leaf"),
        Categorical(["auto", "sqrt", "log2"], name="max_features"),
        Categorical([True, False], name="bootstrap"),
    ]

    # Update default space with kwargs
    updated_space = {**default_space, **kwargs}

    # Convert to skopt space format
    return [updated_space[key] for key in updated_space]


def get_elastic_net_space(**kwargs):
    # Define default space
    default_space = [Real(0.0, 1.0, name="alpha"), Real(0.0, 1.0, name="l1_ratio")]

    # Update default space with kwargs
    updated_space = {**default_space, **kwargs}

    # Convert to skopt space format
    return [updated_space[key] for key in updated_space]


def get_svm_space(**kwargs):
    # Define default space
    default_space = [
        Real(1e-6, 1e6, name="C", prior="log-uniform"),
        Categorical(["linear", "poly", "rbf", "sigmoid"], name="kernel"),
        Real(1e-6, 1e2, name="gamma", prior="log-uniform"),
        Integer(1, 5, name="degree"),
    ]

    # Update default space with kwargs
    updated_space = {**default_space, **kwargs}

    # Convert to skopt space format
    return [updated_space[key] for key in updated_space]


def get_lightgbm_space(**kwargs):
    # Define default space
    default_space = [
        Integer(1, 15, name="num_leaves"),
        Real(0.01, 1.0, name="learning_rate"),
        Integer(1, 100, name="min_data_in_leaf"),
        Real(0.01, 1.0, name="feature_fraction"),
        Real(0.01, 1.0, name="bagging_fraction"),
        Integer(1, 7, name="bagging_freq"),
        Integer(1, 255, name="max_bin"),
    ]

    # Update default space with kwargs
    updated_space = {**default_space, **kwargs}

    # Convert to skopt space format
    return [updated_space[key] for key in updated_space]


def get_xgboost_space(**kwargs):
    # Define default space
    default_space = [
        Integer(1, 15, name="max_depth"),
        Real(0.01, 1.0, name="eta"),
        Real(0.01, 1.0, name="subsample"),
        Real(0.01, 1.0, name="colsample_bytree"),
        Integer(1, 10, name="min_child_weight"),
        Real(0.01, 10.0, name="lambda"),
        Real(0.01, 10.0, name="alpha"),
    ]

    # Update default space with kwargs
    updated_space = {**default_space, **kwargs}

    # Convert to skopt space format
    return [updated_space[key] for key in updated_space]
