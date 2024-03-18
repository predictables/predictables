def model_name_map(name: str) -> str:
    """Map a model name to a canonical name.

    This is useful for users who may refer to the same model by different names.
    The name is standardized, lowercased, and stripped of whitespace. If the name
    is not recognized, an error is raised.
    """
    map = {
        "catboost": "catboost",
        "cb": "catboost",
        "cat": "catboost",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "elastic_net": "elastic_net",
        "elasticnet": "elastic_net",
        "en": "elastic_net",
        "elastic": "elastic_net",
        "lr": "elastic_net",
        "linear_regression": "elastic_net",
        "linearregression": "elastic_net",
        "linear": "elastic_net",
        "logistic_regression": "elastic_net",
        "logisticregression": "elastic_net",
        "logistic": "elastic_net",
        "logit": "elastic_net",
        "logit_regression": "elastic_net",
        "logitregression": "elastic_net",
        "linreg": "elastic_net",
        "logreg": "elastic_net",
        "svm": "svm",
        "svc": "svm",
        "support_vector_machine": "svm",
        "supportvectormachine": "svm",
        "support_vector_classifier": "svm",
        "supportvectorclassifier": "svm",
        "support_vector_regression": "svm",
        "supportvectorregression": "svm",
        "support_vector_regressor": "svm",
        "supportvectorregressor": "svm",
        "svr": "svm",
        "svcr": "svm",
        "svre": "svm",
        "svreg": "svm",
        "sv": "svm",
        "lightgbm": "lightgbm",
        "lgb": "lightgbm",
        "light_gradient_boosting_machine": "lightgbm",
        "lightgradientboostingmachine": "lightgbm",
        "light_gbm": "lightgbm",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "extreme_gradient_boosting": "xgboost",
        "extremegradientboosting": "xgboost",
        "extreme_gradient_boost": "xgboost",
        "extremegradientboost": "xgboost",
        "extreme_gbm": "xgboost",
        "extremegbm": "xgboost",
        "xg": "xgboost",
    }

    name = name.lower().strip()

    if name in map:
        return map[name]
    else:
        raise ValueError(
            f"Invalid model name '{name}'. Must be one of "
            f"'catboost', 'random_forest', 'elastic_net', 'svm', 'lightgbm', "
            f"or 'xgboost'."
        )
