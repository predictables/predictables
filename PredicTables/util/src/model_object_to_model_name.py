from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from .model_name_map import model_name_map


def model_object_to_model_name(model):
    """
    Takes a model object. Returns the name of the model as a string.
    """

    if isinstance(model, CatBoostClassifier) or isinstance(model, CatBoostRegressor):
        out = "catboost"
    elif isinstance(model, RandomForestClassifier) or isinstance(
        model, RandomForestRegressor
    ):
        out = "random_forest"
    elif isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
        out = "lightgbm"
    elif isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
        out = "xgboost"
    elif isinstance(model, SVC) or isinstance(model, SVR):
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
