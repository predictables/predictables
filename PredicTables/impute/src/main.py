import sys

sys.path.append(".")

from initial_impute import initial_impute
from get_cv_folds import get_cv_folds
from train_catboost_model import train_catboost_model
from impute_with_trained_model import impute_with_trained_model
from get_missing_data_mask import get_missing_data_mask


def integrated_imputation_workflow(dataset, learning_rate: float = 0.1):
    """
    This function represents the integrated workflow for imputing missing values using CatBoost.

    :param dataset: The dataset to be imputed.
    :return: The dataset with missing values imputed.
    """
    # Get a missing mask to store which values are missing (and are therefore imputed)
    missing_mask = get_missing_data_mask(dataset)

    # Initial Imputation - will impute missing values with the median
    # or mode depending on the column type
    dataset_imputed = initial_impute(dataset)

    # Generating Cross-Validation Folds
    cv_folds = get_cv_folds(dataset_imputed)

    # Model Training
    trained_models = train_catboost_model(dataset_imputed, cv_folds)

    # Imputation with the Trained Models
    final_dataset = impute_with_trained_model(
        dataset_imputed, missing_mask, trained_models, learning_rate, only_missing=True
    )

    return final_dataset
