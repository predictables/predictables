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
