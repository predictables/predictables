from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from ._model_name_map import model_name_map


def model_object_to_model_name(model):
    """
    Takes a model object. Returns the name of the model as a string.
    """
    if isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        out = "catboost"
    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        out = "random_forest"
    elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
        out = "lightgbm"
    elif isinstance(model, (XGBClassifier, XGBRegressor)):
        out = "xgboost"
    elif isinstance(model, (SVC, SVR)):
        out = "svm"
    elif isinstance(model, ElasticNet):
        out = "elastic_net"
    else:
        raise ValueError(
            f"Invalid model type {type(model)}. Must be one of "
            f"CatBoostClassifier, CatBoostRegressor, "
            f"RandomForestClassifier, RandomForestRegressor, "
            f"LGBMClassifier, LGBMRegressor, "
            f"XGBClassifier, XGBRegressor, "
            f"SVC, SVR, or ElasticNet."
        )

    return model_name_map(out)
