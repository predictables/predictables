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
