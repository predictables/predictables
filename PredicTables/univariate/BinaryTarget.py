import warnings
from collections import namedtuple
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon as js
from scipy.stats import chi2_contingency, entropy, mannwhitneyu, norm, ttest_ind
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    RocCurveDisplay,
    balanced_accuracy_score,
    f1_score,
    hinge_loss,
    log_loss,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import (
    accuracy_score as acc,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from PredicTables.univariate import reconcile_train_test_val_sizes

from PredicTables.univariate.plots import (
    _plot_lift_chart,
    _quintile_lift_plot,
    set_rc_params,
)
from PredicTables.univariate.plots import (
    _rotate_x_labels_if_overlap as rotate_x_lab,
)
from PredicTables.univariate.plots import (
    plot_violin_with_outliers as _plot_violin,
)
from PredicTables.util.stats import gini_coefficient, informedness, kl_divergence
from PredicTables.util import get_column_dtype, to_pl_lf

warnings.filterwarnings("ignore")


cv_fold = namedtuple(
    "cv_fold",
    [
        "train",
        "test",
        "model",
        "param",
        "se",
        "pvalue",
        "train_prob",
        "test_prob",
        "train_y",
        "test_y",
    ],
)
cv_fitted = namedtuple("cv_fit", ["fold", "cv_fold"])
cv_eval = namedtuple(
    "cv_eval",
    [
        "fold",
        "cv_fold",
        "cv_fitted",
        "actselfl_rate",
        "modeled_rate",
        "train_accuracy",
        "val_accuracy",
        "train_roc",
        "val_roc",
        "train_auc",
        "val_auc",
        "train_probabilities",
        "val_probabilities",
    ],
)

cv_container = namedtuple("cv_container", [f"f{i}" for i in range(1, 11)])
cv_idx = namedtuple("cv_idx", ["train", "val"])

rcParams = set_rc_params(rcParams)


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def js_interpretation(js_divergence):
    if js_divergence < 0.05:
        return "Very likely the SAME distribution"
    elif js_divergence < 0.1:
        return "Possibly the SAME distribution"
    elif js_divergence < 0.2:
        return "Unsure whether it is the same distribution or not"
    elif js_divergence < 0.3:
        return "Possibly a DIFFERENT distribution"
    else:
        return "Very likely to be a DIFFERENT distribution"


def generate_distinct_colors(n=30):
    # Define a list of distinct colors
    distinct_colors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#0082c8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#d2f53c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#aa6e28",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000080",
        "#808080",
        "#FFFFFF",
        "#000000",
        "#6a3d9a",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
        "#b2df8a",
    ]
    return distinct_colors[:n]


class Univariate:
    DEFAULT_MODEL_PARAMS = dict(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        penalty="l2",
        cv=5,
        scoring="roc_auc",
    )

    def __init__(
        self,
        df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        feature: str,
        target: str = "evolve_hit_count",
        denom: str = "evolve_quote_count",
        lowest_grain: str = "client_id",
        n_bins: int = 10,
        progressbar: bool = True,
        verbose=False,
        **kwargs,
    ):
        self.df = df
        self.feature = feature
        self.target = target
        self.denom = denom
        self.n_bins = n_bins
        self.lowest_grain = lowest_grain
        self.X_cols = None
        self.progressbar = progressbar

        self.verbose = verbose

        # Get the type of the feature
        self.feature_type = get_column_dtype(
            self.df.select(self.feature).collect()[self.feature]
        )

        # Get the number of unique values for the feature if it is categorical
        if (self.feature_type == "categorical") or (self.feature_type == "binary"):
            self.n_unique = self.df.select(
                [pl.col(self.feature).n_unique().alias(self.feature)]
            )
        else

        # Get the type of the target
        self.target_type = get_column_dtype(
            self.df.select(self.target).collect()[self.target]
        )

        # kwargs overwrite default model params
        self.params = self.DEFAULT_MODEL_PARAMS
        for key, value in kwargs.items():
            self.params[key] = value

        # Initialize the model
        if self.target_type == "binary":
            self.model = LogisticRegressionCV(**self.params)
            self.rf = RandomForestClassifier(**self.params)
        elif self.target_type == "continuous":
            self.model = ElasticNetCV(**self.params)
            self.rf = RandomForestRegressor(**self.params)

        # Initialize the model parameters
        self.params = {}

        # Initialize the auc and sd_auc
        self.mean_auc = None
        self.sd_auc = None

        self.set_up_data()
        self.initialize_metrics_dict()
        self.set_up_pipeline(self.params)
        self.train_model_and_collect_metrics()
        self.set_up_cross_validation()

        self.fit_logistic_regression()

    def reconcile_train_test_val_sizes(self):
        """
        Reconciles the sizes of the training, testing, and validation datasets,
        by either truncating or padding them.

        If any of the train_size, test_size, or val_size attributes are None, they
        will be automatically computed based on the other two attributes, using the
        following rules:
            - If only two attributes are provided, the third one will be set to the
              remaining fraction of the dataset.
            - If only one attribute is provided, the other two will be set to equal
              fractions of the dataset.

        If all three attributes are provided, they will be checked for consistency,
        and an error will be raised if they do not sum up to 1.

        Returns
        -------
        None. The train_size, test_size, and val_size attributes are updated in place.
        """
        self.train_size, self.val_size, self.test_size = reconcile_train_test_val_sizes(
            self.train_size, self.test_size, self.val_size
        )

    def set_up_data(self):
        """
        Sets up the training, validation, and testing datasets for the univariate
        analysis.

        The method performs the following steps:
        1. Groups the input data by the lowest grain (specified by the 'lowest_grain'
            attribute) and gets the maximum target value (either 0 or 1) for each
            group, to identify the groups that have at least one positive target value.
        2. Splits the groups into training and testing sets, using the
            'train_test_split' function from scikit-learn, with the test size specified
            by the 'test_size' attribute.
        3. Splits the training set into training and validation sets, using the same
            method as in step 2, with the validation size specified by the 'val_size'
            attribute.
        4. Extracts the client IDs for the training, validation, and testing sets,
            based on the groups selected in steps 2 and 3.
        5. Extracts the corresponding rows from the input data for the training,
            validation, and testing sets, based on the client IDs.

        The resulting datasets are stored in the 'train', 'val', and 'test' attributes
        of the class instance, respectively.

        Returns
        -------
        None
        """
        # Group by client_id and get the max of the target (either 0 or 1)
        # to get the client_id's that have at least one 1
        lowgrain = (
            self.df[[self.lowest_grain, self.target]]
            .groupby(self.lowest_grain)
            .max()
            .reset_index()
        )

        # Split for train/test sets by client_id (lowest grain)
        self.train_id, self.test_id = train_test_split(
            lowgrain.index,
            test_size=self.test_size,
            random_state=42,
            stratify=lowgrain[self.target],
            shuffle=True,
        )

        # Split for train/val sets by client_id (lowest grain)
        self.train_id, self.val_id = train_test_split(
            self.train_id,
            test_size=(self.val_size / (1 - self.test_size)),
            random_state=42,
            stratify=lowgrain.iloc[self.train_id][self.target],
            shuffle=True,
        )

        # Get the client_id's for the train and test sets
        self.train_client_id = lowgrain.iloc[self.train_id][self.lowest_grain].values
        self.val_client_id = lowgrain.iloc[self.val_id][self.lowest_grain].values
        self.test_client_id = lowgrain.iloc[self.test_id][self.lowest_grain].values

        # Use the client_id's against the original data set to get the train and
        # test sets
        self.train = self.df.loc[self.df[self.lowest_grain].isin(self.train_client_id)][
            [self.feature, self.denom, self.target]
        ].dropna()
        self.val = self.df.loc[self.df[self.lowest_grain].isin(self.val_client_id)][
            [self.feature, self.denom, self.target]
        ].dropna()
        self.test = self.df.loc[self.df[self.lowest_grain].isin(self.test_client_id)][
            [self.feature, self.denom, self.target]
        ].dropna()

        # Make sure there are no client_id's in both train and test sets by
        # taking the intersection of the two sets and making sure the length is 0
        assert len(set(self.train_id).intersection(set(self.test_id))) == 0, f"There are client_id's in both train and test: \
                \n  {set(self.train_id).intersection(set(self.test_id))}"

        # Make sure there are no client_id's in both train and val sets by
        # taking the intersection of the two sets and making sure the length is 0
        assert len(set(self.train_id).intersection(set(self.val_id))) == 0, f"There are client_id's in both train and val: \
                \n  {set(self.train_id).intersection(set(self.val_id))}"

        # Make sure there are no client_id's in both val and test sets by
        # taking the intersection of the two sets and making sure the length is 0
        assert len(set(self.val_id).intersection(set(self.test_id))) == 0, f"There are client_id's in both val and test: \
                \n  {set(self.val_id).intersection(set(self.test_id))}"

    def set_up_cross_validation(self):
        """
        Sets up the cross-validation for the univariate analysis.
        Creates a series of integers representing the folds, where the series
        is the same length as the training set.
        """
        # Get the validation indices for the training set
        val = self.cv_idx.val

        # Loop through the validation indices, and assign to
        # each row of the training set the fold number
        # corresponding to the validation index
        self.fold = pd.Series([-1] * len(self.train), index=self.train.index)
        for i, fold in enumerate(val):
            # print(f"Fold {i + 1}: {len(fold)}\n{fold}")

            self.fold[self.train.reset_index().index.to_series().isin(fold).values] = (
                i + 1
            )

    ## Getters
    def GetTrain(
        self,
        fold: Optional[int] | Optional[Callable] = None,
        feature: Optional[str]
        | Optional[float]
        | Optional[int]
        | Optional[Callable] = None,
        target: Optional[str]
        | Optional[float]
        | Optional[int]
        | Optional[Callable] = None,
        denom: Optional[str]
        | Optional[float]
        | Optional[int]
        | Optional[Callable] = None,
        unique_lowest_grain: bool = False,
        na_rm: bool = False,
        standardize: bool = True,
        return_std_params: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the training dataset. Optionally filters by fold number and/or
        removes rows with missing values.
        """
        df = self.train.copy()[[self.target, self.feature, self.denom]]
        df["fold"] = self.fold
        df["feature"] = df[self.feature]
        df["target"] = df[self.target]
        df["denom"] = df[self.denom]
        if self.type == "categorical":
            df = pd.concat([df, self.GetDummies("train")], axis=1)
        if fold is not None:
            if isinstance(fold, int):
                df = df.loc[df["fold"] == fold]
            elif callable(fold):
                df = df.loc[df["fold"].apply(fold)]

        if na_rm:
            df.dropna(inplace=True)

        if feature is not None:
            if isinstance(feature, str):
                df = df.loc[df[self.feature] == feature]
            elif isinstance(feature, float):
                df = df.loc[df[self.feature] == feature]
            elif isinstance(feature, int):
                df = df.loc[df[self.feature] == feature]
            elif callable(feature):
                df = df.loc[df[self.feature].apply(feature)]

        if target is not None:
            if isinstance(target, str):
                df = df.loc[df[self.target] == target]
            elif isinstance(target, float):
                df = df.loc[df[self.target] == target]
            elif isinstance(target, int):
                df = df.loc[df[self.target] == target]
            elif callable(target):
                df = df.loc[df[self.target].apply(target)]

        if denom is not None:
            if isinstance(denom, str):
                df = df.loc[df[self.denom] == denom]
            elif isinstance(denom, float):
                df = df.loc[df[self.denom] == denom]
            elif isinstance(denom, int):
                df = df.loc[df[self.denom] == denom]
            elif callable(denom):
                df = df.loc[df[self.denom].apply(denom)]

        if unique_lowest_grain:
            df["lowest_grain"] = self.df.loc[
                self.df.index.to_series().isin(self.train.index.tolist())
            ][self.lowest_grain].values

            # get max target for each lowest grain to ensure you get a hit if there
            # is at least one
            max_target = (
                df.groupby("lowest_grain")[self.target]
                .max()
                .reset_index()
                .drop_duplicates()
            )

            # join back to the df to get the feature and denom corresponding to the
            # max target
            df = df.merge(max_target, on="lowest_grain", how="inner")

            # drop the max target column
            df.drop(columns=self.target + "_y", inplace=True)

            # rename the max target column to target
            df.rename(columns={self.target + "_x": self.target}, inplace=True)

            # drop duplicates
            df.drop_duplicates(inplace=True)

        if standardize:
            if self.type == "continuous":
                mu = df[self.feature].mean()
                sigma = df[self.feature].std()
                df[self.feature] = (df[self.feature] - mu) / sigma

        if return_std_params:
            return df, mu, sigma
        else:
            return df

    def GetVal(
        self,
        feature: Optional[str]
        | Optional[float]
        | Optional[int]
        | Optional[Callable] = None,
        target: Optional[str]
        | Optional[float]
        | Optional[int]
        | Optional[Callable] = None,
        denom: Optional[str]
        | Optional[float]
        | Optional[int]
        | Optional[Callable] = None,
        na_rm: bool = False,
        standardize: bool = True,
    ) -> pd.DataFrame:
        """
        Returns the validation dataset. Optionally removes rows with missing values.
        """
        df = self.val.copy()[[self.target, self.feature, self.denom]]
        df["feature"] = df[self.feature]
        df["target"] = df[self.target]
        df["denom"] = df[self.denom]

        if self.type == "categorical":
            df = pd.concat([df, self.GetDummies("val")], axis=1)

        if na_rm:
            df.dropna(inplace=True)

        if feature is not None:
            if isinstance(feature, str):
                df = df.loc[df[self.feature] == feature]
            elif isinstance(feature, float):
                df = df.loc[df[self.feature] == feature]
            elif isinstance(feature, int):
                df = df.loc[df[self.feature] == feature]
            elif callable(feature):
                df = df.loc[df[self.feature].apply(feature)]

        if target is not None:
            if isinstance(target, str):
                df = df.loc[df[self.target] == target]
            elif isinstance(target, float):
                df = df.loc[df[self.target] == target]
            elif isinstance(target, int):
                df = df.loc[df[self.target] == target]
            elif callable(target):
                df = df.loc[df[self.target].apply(target)]

        if denom is not None:
            if isinstance(denom, str):
                df = df.loc[df[self.denom] == denom]
            elif isinstance(denom, float):
                df = df.loc[df[self.denom] == denom]
            elif isinstance(denom, int):
                df = df.loc[df[self.denom] == denom]
            elif callable(denom):
                df = df.loc[df[self.denom].apply(denom)]

        if standardize:
            if self.type == "continuous":
                _, mu, sigma = self.GetTrain(return_std_params=True)
                df[self.feature] = (df[self.feature] - mu) / sigma

        return df

    def GetDummies(self, train_val_test="train"):
        if train_val_test == "train":
            df = self.train.copy()
        elif train_val_test == "val":
            df = self.val.copy()
        elif train_val_test == "test":
            df = self.test.copy()

        if self.type == "categorical":
            out = pd.get_dummies(
                df[[self.feature]], columns=[self.feature], drop_first=True
            ).astype(int)

            out.columns = [
                col.replace(" ", "_").replace("-", "_").lower() for col in out.columns
            ]
            self.X_cols = out.columns.tolist()
        else:
            self.X_cols = [self.feature]
        return out

    def initialize_metrics_dict(self):
        """
        Initialize metrics dictionaries for evaluation.
        """
        # Initialize the fitted dictionary and the evaluation dictionary
        self.fitted = {}
        self.fitted["eval"] = {}
        self.fitted["eval"]["fold"] = []
        self.fitted["eval"]["actselfl_rate"] = []
        self.fitted["eval"]["modeled_rate"] = []
        self.fitted["eval"]["train_accuracy"] = []
        self.fitted["eval"]["val_accuracy"] = []
        self.fitted["eval"]["train_roc"] = []
        self.fitted["eval"]["val_roc"] = []
        self.fitted["eval"]["train_auc"] = []
        self.fitted["eval"]["val_auc"] = []
        self.fitted["eval"]["train_probabilities"] = []
        self.fitted["eval"]["val_probabilities"] = []
        self.fitted["eval"]["coef"] = []

        self.fit_obj = namedtuple(
            "fit",
            [
                "model",
                "params",
                "pvalues",
                "aic",
                "bic",
                "se",
                "conf_int",
                "deviance",
                "fittedvalues",
                "llf",
                "hat_matrix_diag",
                "influence",
                "information_criteria",
                "nobs",
                "resid_pearson",
                "resid_deviance",
                "resid_anscombe",
                "resid_response",
                "resid_working",
                "summary",
                "X",
                "y",
                "yhat",
            ],
        )
        self.fit = None

    def fit_logistic_regression(self):
        """
        Fits a logistic regression model using statsmodels.

        If the feature is binary, starting values are passed based on the proportion
        of 1's in the feature. A very small jitter is added to help with convergence.
        """
        # if the feature is binary, can pass starting values based on the
        # proportion of 1's in the feature
        if self.is_binary:
            # Pass in starting values that are the logit transform of the proportion
            # of 1's in the feature
            prop_one = np.mean(self.GetTrain(feature=1)["target"])

            # add a very small jitter to help with convergence
            prop_one = prop_one + 1e-6
            logit_one = [np.log(prop_one / (1 - prop_one))]

            if self.type == "categorical":
                X = self.GetTrain().feature.values.reshape(-1, 1)
                X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
                logit_one = np.concatenate([np.zeros(1), logit_one], axis=0)
            else:
                X = sm.add_constant(self.GetTrain().feature)

            from numpy.linalg import matrix_rank

            # X = sm.add_constant(self.train[self.feature])
            if self.verbose:
                print("Rank of the matrix:", matrix_rank(X.values))

            model = smf.glm(
                formula=f"Q('{self.target}') ~ Q('{self.feature}')",
                data=self.train,
                family=sm.families.Binomial(),
            ).fit()
        else:
            # fit the model
            model = smf.glm(
                formula=f"Q('{self.target}') ~ Q('{self.feature}')",
                data=self.train,
                family=sm.families.Binomial(),
            ).fit()

        # get the fitted values
        self.fit = self.fit_obj(
            model,
            model.params,
            model.pvalues,
            model.aic,
            model.bic,
            model.bse,
            model.conf_int(),
            model.deviance,
            model.fittedvalues,
            model.llf,
            model.get_hat_matrix_diag(),
            model.get_influence(),
            model.info_criteria,
            model.nobs,
            model.resid_pearson,
            model.resid_deviance,
            model.resid_anscombe,
            model.resid_response,
            model.resid_working,
            model.summary(),
            self.train[[self.feature]],
            self.train[[self.target]],
            model.fittedvalues,
        )

    def set_up_pipeline(self, params):
        """
        Initialize the pipeline for transforming data and training the model.
        Merge passed params with default params.
        """
        # Default parameters for the logistic regression model
        default_params = self.DEFAULT_MODEL_PARAMS

        # Merge the default parameters with the passed parameters
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

        # Initialize the pipeline
        if self.type == "categorical":
            self.preprocess = ("onehot", OneHotEncoder(handle_unknown="ignore"))
        elif self.type == "continuous":
            self.preprocess = ("scale", StandardScaler())
        else:
            raise ValueError(f"Type {self.type} is not supported.")
        self.pipeline = Pipeline(
            [self.preprocess, ("model", LogisticRegressionCV(**self.params))]
        )

    def _train_model(self, train_data, val_data):
        # Train pipeline on the training data
        model = self.pipeline.fit(train_data[[self.feature]], train_data[self.target])

        # Get predicted probabilities for train and validation data
        train_prob = model.predict_proba(train_data[[self.feature]])[:, 1]
        val_prob = model.predict_proba(val_data[[self.feature]])[:, 1]

        # Get actselfl values for train and validation data
        train_y = train_data[self.target].values
        val_y = val_data[self.target].values

        return model, train_prob, val_prob, train_y, val_y

    def train_model_and_collect_metrics(self):
        """
        Train the model and collect metrics using k-fold cross-validation.
        """
        # Loop through each fold using tqdm for progress indication
        trains, vals = [], []
        if self.progressbar:
            for fold, (train_idx, val_idx) in tqdm(
                enumerate(self.kfold.split(self.train, self.train[self.target])),
                total=self.n_bins,
            ):
                trains.append(train_idx)
                vals.append(val_idx)

                # Train and evaluate the model for the current fold
                self.train_and_evaluate_for_fold(fold, train_idx, val_idx)
        else:
            for fold, (train_idx, val_idx) in enumerate(
                self.kfold.split(self.train, self.train[self.target])
            ):
                trains.append(train_idx)
                vals.append(val_idx)

                # Train and evaluate the model for the current fold
                self.train_and_evaluate_for_fold(fold, train_idx, val_idx)

        self.cv_idx = cv_idx(trains, vals)

        # Once you have trained and evalselfted the model for each fold,
        # train the model on the entire training set and validate on the
        # held-out validation set
        (
            self.model_train,
            self.fitted_train_prob,
            self.fitted_val_prob,
            self.y_train,
            self.y_val,
        ) = self._train_model(self.train, self.val)

    def train_and_evaluate_for_fold(self, fold, train_idx, val_idx):
        """
        For a given fold, execute the training and evaluation steps.
        Save the model, predictions, and metrics.
        """
        # Get training and validation data for the current fold
        train_data = self.train.iloc[train_idx]
        val_data = self.train.iloc[val_idx]

        # Initialize the fitted dictionary for the current fold
        self.fitted[fold] = {}
        self.fitted[fold]["train"] = {}
        self.fitted[fold]["val"] = {}

        # Train pipeline on the training data
        model, train_prob, val_prob, train_y, val_y = self._train_model(
            train_data, val_data
        )

        # Store the model in the fitted dictionary
        self.fitted[fold]["pipeline"] = model
        self.fitted[fold]["train"]["prob"] = train_prob
        self.fitted[fold]["val"]["prob"] = val_prob
        self.fitted[fold]["train"][self.target] = train_y
        self.fitted[fold]["val"][self.target] = val_y

        # Calculate and store metrics
        self.calculate_and_store_metrics(fold)

    def calculate_and_store_metrics(self, fold):
        """
        Calculate metrics like accuracy, ROC-AUC for a specific fold.
        Store them in the metrics dictionary.
        """
        # Compute metrics for train set
        # train_accuracy = acc(self.GetTrain(fold=fold)).target.values,
        #                      self.fitted[fold]['train']['prob'] > 0.5)
        train_accuracy = acc(
            self.fitted[fold]["train"][self.target],
            self.fitted[fold]["train"]["prob"] > 0.5,
        )
        train_roc = roc_curve(
            self.fitted[fold]["train"][self.target], self.fitted[fold]["train"]["prob"]
        )
        train_auc = roc_auc_score(
            self.fitted[fold]["train"][self.target], self.fitted[fold]["train"]["prob"]
        )

        # Compute metrics for validation set
        val_accuracy = acc(
            self.fitted[fold]["val"][self.target],
            self.fitted[fold]["val"]["prob"] > 0.5,
        )
        val_roc = roc_curve(
            self.fitted[fold]["val"][self.target], self.fitted[fold]["val"]["prob"]
        )
        val_auc = roc_auc_score(
            self.fitted[fold]["val"][self.target], self.fitted[fold]["val"]["prob"]
        )

        # Compute fitted coefficients
        coef = self.fitted[fold]["pipeline"]["model"].coef_[0][0]

        # Store metrics in the fitted dictionary
        self.fitted["eval"]["fold"].append(fold)
        self.fitted["eval"]["train_accuracy"].append(train_accuracy)
        self.fitted["eval"]["val_accuracy"].append(val_accuracy)
        self.fitted["eval"]["train_roc"].append(train_roc)
        self.fitted["eval"]["val_roc"].append(val_roc)
        self.fitted["eval"]["train_auc"].append(train_auc)
        self.fitted["eval"]["val_auc"].append(val_auc)
        self.fitted["eval"]["train_probabilities"].append(
            self.fitted[fold]["train"]["prob"]
        )
        self.fitted["eval"]["val_probabilities"].append(
            self.fitted[fold]["val"]["prob"]
        )
        self.fitted["eval"]["coef"].append(coef)

    

    

    

    def plot_chi_sqselfred_test(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
        ct = pd.crosstab(self.train[self.feature], self.train[self.target])
        chi2, p, _, _ = chi2_contingency(ct)
        chi2_results_msg = f"Chi-Sqselfred Test: chi2 = {chi2:.4f}, p-value = {p:.4f}"
        significance_message = (
            f"{self._get_significance_band(p, 'indicated association')}"
        )

        # Plot the heatmap, as a percentage of the row total
        sns.heatmap(
            ct / ct.sum(axis=1).values.reshape(-1, 1),
            annot=True,
            fmt=",",
            cmap="viridis",
            ax=ax,
        )
        ax.set_title(
            f"Chi-Sqselfred Test for Independence\n{chi2_results_msg}\n\
    {significance_message}"
        )

        ax = rotate_x_lab(ax)
        ax.figure.tight_layout()
        return ax

    def plot_point_plot(self, ax=None):
        df = self.GetTrain()[[self.feature, self.target]]

        if self.type == "categorical":
            df[self.feature] = df[self.feature].astype(str)

        # Compute ratio and sort
        ratio_df = (
            df.groupby(self.feature)
            .apply(lambda x: x[self.target].mean())
            .reset_index(name="Ratio")
        )
        sorted_features = ratio_df.sort_values("Ratio", ascending=False)[self.feature]

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        # Loop through each fold, get the self.target, self.denom, and
        # self.feature columns
        fold_df_list = []
        lines = []
        all_features = sorted_features.unique()
        for i in range(1, self.n_bins + 1):
            # Create 1-row DF for the cumulative target / denom for the current
            # fold
            tempdf = (
                pd.DataFrame(
                    {
                        self.feature: self.GetTrain(fold=i)[self.feature],
                        self.denom: self.GetTrain(fold=i)[self.denom],
                        self.target: self.GetTrain(fold=i)[self.target],
                    }
                )
                .groupby(self.feature)
                .agg({self.target: "sum", self.denom: "sum"})
                .assign(ratio=lambda x: x[self.target] / x[self.denom])
                .assign(fold=i)
                .reset_index()
            )

            # Find the missing features in the current fold
            missing_features = set(all_features) - set(tempdf[self.feature])

            # If there are missing features, pad tempdf with zeros for those features
            if missing_features:
                pad_df = pd.DataFrame(
                    {
                        self.feature: list(missing_features),
                        self.target: [0] * len(missing_features),
                        self.denom: [0] * len(missing_features),
                        "fold": f"{i}" * len(missing_features),
                        "ratio": [0] * len(missing_features),
                    }
                )
                tempdf = pd.concat([tempdf, pad_df], ignore_index=True)

            # print(f"tempdf: {tempdf.head()}")
            fold_df_list.append(tempdf)

            # Sort tempdf based on sorted_features
            tempdf.set_index(self.feature, inplace=True)
            tempdf = tempdf.loc[sorted_features]
            tempdf.reset_index(inplace=True)

            # Plot the individual fold ratios
            line = ax.plot(
                tempdf[self.feature],
                tempdf["ratio"],
                color="gray",
                alpha=0.3,
                marker="x",
            )
            #    linestyle='dotted')
            lines.append(line)

        fold_df = pd.concat(fold_df_list).reset_index()

        # Check if the feature exists in fold_df
        if self.feature not in fold_df.columns:
            print(fold_df.head())
            raise ValueError(
                f"The feature {self.feature} is not present in fold_df. \
Available columns are {fold_df.columns.tolist()}"
            )

        # Calculate the average ratio by feature
        avg_ratio = (
            fold_df.groupby(self.feature)["ratio"].mean().reset_index(name="ratio")
        )

        # Calculate std dev of the ratio by feature
        std_ratio = fold_df.groupby(self.feature)["ratio"].std().reset_index(name="sd")

        # Sort avg_ratio by Ratio
        avg_ratio = avg_ratio.rename(columns=dict(ratio="Ratio")).merge(
            std_ratio, on=self.feature, how="left"
        )

        # Sort the avg_ratio DataFrame based on the sorted_features
        avg_ratio.set_index(self.feature, inplace=True)
        avg_ratio = avg_ratio.loc[sorted_features]
        avg_ratio.reset_index(inplace=True)

        # Plot average ratio for each level of the feature
        ax.plot(
            avg_ratio[self.feature],
            avg_ratio["Ratio"],
            color="black",
            marker="o",
            label="Average Ratio",
        )

        # Add +/- 1 standard deviation region to the plot
        ax.fill_between(
            avg_ratio[self.feature],
            avg_ratio["Ratio"] - avg_ratio["sd"],
            avg_ratio["Ratio"] + avg_ratio["sd"],
            color="gray",
            alpha=0.2,
            label="Standard Deviation",
        )

        # Plot a horizontal line at the overall average ratio
        ax.axhline(
            y=self.train[self.target].sum() / self.train[self.denom].sum(),
            color="red",
            linestyle="--",
            label="Overall Average Ratio",
        )

        # Set x and y labels + title
        ax.set_title(
            f"Relationship of [{self._plot_label(self.feature)}] with ratio \
of\n[{self._plot_label(self.target)}] / [{self._plot_label(self.denom)}]"
        )
        ax.set_ylabel(
            f"[{self._plot_label(self.target)}] / \
[{self._plot_label(self.denom)}]"
        )

        # Set x ticks and labels
        ax = rotate_x_lab(ax)
        ax.figure.tight_layout()

        # Grab the lines for legend
        lines, labels = ax.get_legend_handles_labels()

        # Keep the last three lines (Average Ratio, Standard Deviation,
        # Overall Average Ratio)
        ax.legend(
            [lines[-3], lines[-2], lines[-1]],
            ["Average Ratio", "Ave +/- 1 SD", "Overall Average Ratio"],
            fontsize=16,
        )

        return ax

    def plotly_point_plot(self):
        df = self.GetTrain()[[self.feature, self.target]]

        if self.type == "categorical":
            df[self.feature] = df[self.feature].astype(str)

        # Compute ratio and sort
        ratio_df = (
            df.groupby(self.feature)
            .apply(lambda x: x[self.target].mean())
            .reset_index(name="Ratio")
        )
        sorted_features = ratio_df.sort_values("Ratio", ascending=False)[self.feature]

        # Initialize figure
        fig = go.Figure()

        all_features = sorted_features.unique()
        fold_df_list = []

        # Loop through each fold, plot the individual fold ratios
        for i in range(1, self.n_bins + 1):
            tempdf = (
                pd.DataFrame(
                    {
                        self.feature: self.GetTrain(fold=i)[self.feature],
                        self.denom: self.GetTrain(fold=i)[self.denom],
                        self.target: self.GetTrain(fold=i)[self.target],
                    }
                )
                .groupby(self.feature)
                .agg({self.target: "sum", self.denom: "sum"})
                .assign(ratio=lambda x: x[self.target] / x[self.denom])
                .assign(fold=i)
                .reset_index()
            )

            missing_features = set(all_features) - set(tempdf[self.feature])
            if missing_features:
                pad_df = pd.DataFrame(
                    {
                        self.feature: list(missing_features),
                        self.target: [0] * len(missing_features),
                        self.denom: [0] * len(missing_features),
                        "fold": i,
                        "ratio": [0] * len(missing_features),
                    }
                )
                tempdf = pd.concat([tempdf, pad_df], ignore_index=True)

            fold_df_list.append(tempdf)

            # Sort tempdf based on sorted_features
            tempdf.set_index(self.feature, inplace=True)
            tempdf = tempdf.loc[sorted_features]
            tempdf.reset_index(inplace=True)

            fig.add_trace(
                go.Scatter(
                    x=tempdf[self.feature],
                    y=tempdf["ratio"],
                    mode="markers",
                    marker=dict(color="gray", size=8),
                    name=f"Fold {i}",
                )
            )

        fold_df = pd.concat(fold_df_list).reset_index()

        # Calculate the average ratio and standard deviation by feature
        avg_ratio = (
            fold_df.groupby(self.feature)["ratio"]
            .mean()
            .reset_index(name="Average Ratio")
        )
        std_ratio = (
            fold_df.groupby(self.feature)["ratio"]
            .std()
            .reset_index(name="Standard Deviation")
        )

        # Merge average ratio and standard deviation data
        avg_ratio = avg_ratio.merge(std_ratio, on=self.feature, how="left")

        # Sort the avg_ratio DataFrame based on the sorted_features
        avg_ratio.set_index(self.feature, inplace=True)
        avg_ratio = avg_ratio.loc[sorted_features]
        avg_ratio.reset_index(inplace=True)

        # Plot average ratio for each level of the feature
        fig.add_trace(
            go.Scatter(
                x=avg_ratio[self.feature],
                y=avg_ratio["Average Ratio"],
                mode="markers+lines",
                marker=dict(color="black", size=10),
                name="Average Ratio",
            )
        )

        # Add +/- 1 standard deviation region to the plot
        fig.add_trace(
            go.Scatter(
                x=avg_ratio[self.feature],
                y=avg_ratio["Average Ratio"] + avg_ratio["Standard Deviation"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=avg_ratio[self.feature],
                y=avg_ratio["Average Ratio"] - avg_ratio["Standard Deviation"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(width=0),
                name="Standard Deviation",
            )
        )

        # Plot a horizontal line at the overall average ratio
        overall_avg_ratio = (
            self.GetTrain()[self.target].sum() / self.GetTrain()[self.denom].sum()
        )
        fig.add_hline(
            y=overall_avg_ratio,
            line=dict(color="red", dash="dash"),
            name="Overall Average Ratio",
        )

        # Update the layout
        fig.update_layout(
            title=f"Relationship of {self._plot_label(self.feature)} with ratio of {self._plot_label(self.target)} / {self._plot_label(self.denom)}",
            xaxis_title=self._plot_label(self.feature),
            yaxis_title=f"Ratio of {self._plot_label(self.target)} / {self._plot_label(self.denom)}",
            legend_title="Legend",
            xaxis=dict(tickvals=sorted_features, ticktext=sorted_features),
            yaxis=dict(tickformat=".1%"),
        )

        return fig

    def plot_box_plot(self, ax=None):
        df = self.train[[self.feature, self.target]]

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        sns.boxplot(data=df, x=self.target, y=self.feature, ax=ax)

        ax.set_title(
            f"Distribution of [{self._plot_label(self.feature)}] by \
[{self._plot_label(self.target)}]"
        )
        ax.set_xlabel(self._plot_label(self.target))
        ax.set_ylabel(self._plot_label(self.feature))

        ax = rotate_x_lab(ax)
        ax.figure.tight_layout()

        return ax

    def plot_density_plot(self, significance_level=0.05, ax=None, opacity=0.5):
        df = self.train[[self.feature, self.target]]

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        unique_targets = df[self.target].unique()

        data_by_target = {}
        for target_val in unique_targets:
            sns.kdeplot(
                df[df[self.target] == target_val][self.feature],
                ax=ax,
                label=f"{self._plot_label(self.target)} = {target_val}",
                alpha=opacity,
                fill=True,
            )
            data_by_target[target_val] = df[df[self.target] == target_val][self.feature]

        # Mann-Whitney U Test
        u_stat, p_value = mannwhitneyu(
            data_by_target[unique_targets[0]], data_by_target[unique_targets[1]]
        )

        # Print the results of the Mann-Whitney U Test
        subtitle_message = (
            f"\nDistributions are the same at the {significance_level:.0%} level"
            if p_value > significance_level
            else f"\nDistributions are different at the {significance_level:.0%} level"
        )

        ax.set_title(
            f"Density Plot of [{self._plot_label(self.feature)}] by \
[{self._plot_label(self.target)}]{subtitle_message}"
        )

        annotation_text = f"Mann-Whitney\nU-Test Statistic:\n{u_stat:.1f}\n\np-value:\n\
{p_value:.2f}"
        ax.annotate(
            annotation_text,
            xy=(0.7375, 0.625),
            xycoords="axes fraction",
            fontsize=16,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="aliceblue",
                alpha=0.5,
            ),
        )

        ax.set_xlabel(self._plot_label(self.feature))
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=16)

        ax = rotate_x_lab(ax)
        ax.figure.tight_layout()
        return ax

    def plotly_density_plot(self, significance_level=0.05, opacity=0.5):
        df = self.train[[self.feature, self.target]]
        unique_targets = df[self.target].unique()

        fig = go.Figure()

        data_by_target = {}
        for target_val in unique_targets:
            # Filter the data for the current target value
            filtered_data = df[df[self.target] == target_val][self.feature]
            data_by_target[target_val] = filtered_data

            # Add a KDE trace for the current target value
            fig.add_trace(
                go.Histogram(
                    x=filtered_data,
                    histnorm="probability density",
                    opacity=opacity,
                    name=f"{self._plot_label(self.target)} = {target_val}",
                )
            )

        # Perform Mann-Whitney U Test between the two groups
        u_stat, p_value = mannwhitneyu(
            data_by_target[unique_targets[0]], data_by_target[unique_targets[1]]
        )

        # Determine the subtitle message based on p_value
        subtitle_message = (
            f"Distributions are the same at the {significance_level:.0%} significance level"
            if p_value > significance_level
            else f"Distributions are different at the {significance_level:.0%} significance level"
        )

        # Add annotations with the Mann-Whitney U Test results
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"Mann-Whitney U-Test Statistic: {u_stat:.1f}<br>p-value: {p_value:.2f}",
            showarrow=False,
            font=dict(size=16),
            align="right",
            bgcolor="aliceblue",
            opacity=0.8,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
        )

        # Update the layout of the figure
        fig.update_layout(
            title=f"Density Plot of {self._plot_label(self.feature)} by {self._plot_label(self.target)}<br><sup>{subtitle_message}</sup>",
            xaxis_title=self._plot_label(self.feature),
            yaxis_title="Density",
            barmode="overlay",  # Overlay the KDE plots
            legend_title_text="Legend",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def _calculate_empirical_cdf(self, fold_idx=None):
        # Extract train data for this fold
        if fold_idx is None:
            fold_data = self.train
        else:
            fold_data = self.train.loc[fold_idx]

        # Compute empirical CDF
        sorted_data = fold_data[self.feature].sort_values()
        normalized_data = (sorted_data - sorted_data.min()) / (
            sorted_data.max() - sorted_data.min()
        )
        empirical_cdf = normalized_data.cumsum() / normalized_data.sum()

        # Interpolate empirical CDF to a common grid
        min_x, max_x = sorted_data.min(), sorted_data.max()
        common_grid = np.linspace(min_x, max_x, 500)
        interpolator = interp1d(
            sorted_data, empirical_cdf, bounds_error=False, fill_value=(0, 1)
        )
        resampled_cdf = interpolator(common_grid)

        return common_grid, resampled_cdf

    def plot_cdf(self, ax=None):
        df = self.train[[self.feature, self.target]]

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        unique_targets = df[self.target].unique()
        resampled_cdfs = {}

        # Plot the cross-validated CDF functions for each target value & fold
        for fold in range(self.n_bins):
            folddf = (
                self.train[[self.feature, self.target]]
                .iloc[self.cv_idx.val[fold]]
                .loc[self.train[self.feature].ne(-0.01)]
            )

            for target_val in unique_targets:
                target_idx = folddf.loc[folddf[self.target] == target_val].index

                # Using the private method to calculate empirical CDF
                common_grid, resampled_cdf = self._calculate_empirical_cdf(target_idx)

                ax.plot(
                    common_grid,
                    resampled_cdf,
                    color=("blue" if target_val == 0 else "orange"),
                    alpha=0.3,
                )

                resampled_cdfs[target_val] = resampled_cdf

        # Plot the actual empirical CDF functions
        for target_val in unique_targets:
            target_idx = (
                df.loc[df[self.target] == target_val]
                .loc[df[self.feature].ne(-0.01)]
                .index
            )

            # Using the private method to calculate empirical CDF
            common_grid, resampled_cdf = self._calculate_empirical_cdf(target_idx)

            ax.plot(
                common_grid,
                resampled_cdf,
                color=("blue" if target_val == 0 else "orange"),
                label=f"{self._plot_label(self.target)} = {target_val}",
            )

            resampled_cdfs[target_val] = resampled_cdf

        # Calculate JS Divergence
        js_divergence = js(
            resampled_cdfs[unique_targets[0]], resampled_cdfs[unique_targets[1]]
        )
        interpretation = js_interpretation(js_divergence)

        # Add in scatter points for the actual data (again blue for 0, orange for 1)
        for target_val in unique_targets:
            target_idx = (
                df.loc[df[self.target] == target_val]
                .loc[df[self.feature].ne(-0.01)]
                .index
            )

            # calculate jitter term
            jitter = 0.01 * np.random.randn(
                self.train[self.target].eq(target_val).sum()
            )
            ax.scatter(
                self.train.loc[
                    self.train[self.target].eq(target_val), self.feature
                ].loc[self.train[self.target].ne(-0.01)]
                + jitter,
                self.train.loc[self.train[self.target].eq(target_val), self.target].loc[
                    self.train[self.target].ne(-0.01)
                ]
                + jitter,
                marker="|",
                edgecolors=("blue" if target_val == 0 else "orange"),
                color=("blue" if target_val == 0 else "orange"),
                alpha=0.2,
            )

        # Annotating the plot
        significance_message = f"Jensen-Shannon\nDivergence: {js_divergence:.2f}\nInterpretation:\n{interpretation}"
        subtitle_message = (
            f"\nInterpreted based on Jenson-Shannon Divergence:\n{interpretation}"
        )

        ax.annotate(
            significance_message,
            xy=(0.55, 0.15),
            xycoords="axes fraction",
            fontsize=16,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="aliceblue",
                alpha=0.5,
            ),
        )

        ax.set_title(
            f"CDF of [{self._plot_label(self.feature)}] by \
[{self._plot_label(self.target)}]{subtitle_message}",
            loc="center",
        )
        ax.legend(loc="lower right", fontsize=16)
        ax.set_xlabel(self._plot_label(self.feature))
        ax.set_ylabel("Empirical Cumulative Distribution Function")

        ax = rotate_x_lab(ax)
        ax.figure.tight_layout()

        return ax

    def plotly_cdf(self):
        df = self.train[[self.feature, self.target]]

        unique_targets = df[self.target].unique()
        resampled_cdfs = {}
        fig = go.Figure()

        # Plot the cross-validated CDF functions for each target value & fold
        for fold in range(self.n_bins):
            folddf = (
                self.train[[self.feature, self.target]]
                .iloc[self.cv_idx.val[fold]]
                .loc[self.train[self.feature].ne(-0.01)]
            )

            for target_val in unique_targets:
                target_idx = folddf.loc[folddf[self.target] == target_val].index

                # Using the private method to calculate empirical CDF
                common_grid, resampled_cdf = self._calculate_empirical_cdf(target_idx)

                fig.add_trace(
                    go.Scatter(
                        x=common_grid,
                        y=resampled_cdf,
                        line=dict(
                            color=("blue" if target_val == 0 else "orange"), width=1
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                        mode="lines",
                        opacity=0.3,  # alpha
                    )
                )

                resampled_cdfs[target_val] = resampled_cdf

        # Plot the actual empirical CDF functions
        for target_val in unique_targets:
            target_idx = (
                df.loc[df[self.target] == target_val]
                .loc[df[self.feature].ne(-0.01)]
                .index
            )

            # Using the private method to calculate empirical CDF
            common_grid, resampled_cdf = self._calculate_empirical_cdf(target_idx)

            fig.add_trace(
                go.Scatter(
                    x=common_grid,
                    y=resampled_cdf,
                    line=dict(color=("blue" if target_val == 0 else "orange"), width=2),
                    name=f"{self._plot_label(self.target)} = {target_val}",
                    mode="lines",
                )
            )

            resampled_cdfs[target_val] = resampled_cdf

        # Calculate JS Divergence here and interpretation
        js_divergence = js(
            resampled_cdfs[unique_targets[0]], resampled_cdfs[unique_targets[1]]
        )
        interpretation = js_interpretation(js_divergence)

        # Add in scatter points for the actual data (again blue for 0, orange for 1)
        for target_val in unique_targets:
            jitter = 0.01 * np.random.randn(len(df[df[self.target] == target_val]))
            fig.add_trace(
                go.Scatter(
                    x=df[df[self.target] == target_val][self.feature] + jitter,
                    y=np.random.uniform(
                        low=0, high=1, size=len(df[df[self.target] == target_val])
                    )
                    * jitter,
                    mode="markers",
                    marker=dict(
                        color=("blue" if target_val == 0 else "orange"), size=3
                    ),
                    name=f"{self._plot_label(self.target)} = {target_val} (data)",
                    opacity=0.2,  # alpha
                )
            )

        # Add a title with JS Divergence information
        # Use `text` parameter to include additional subtitle information
        fig.update_layout(
            title=f"CDF of {self._plot_label(self.feature)} by {self._plot_label(self.target)}",
            xaxis_title=self._plot_label(self.feature),
            yaxis_title="Empirical Cumulative Distribution Function",
            legend_title_text="Legend",
            hovermode="closest",
        )

        # Add annotations for JS Divergence information
        fig.add_annotation(
            x=0.95,
            y=0.05,
            xanchor="right",
            yanchor="bottom",
            text=f"Jensen-Shannon Divergence: {js_divergence:.2f}<br>Interpretation: {interpretation}",
            showarrow=False,
            bgcolor="aliceblue",
            opacity=0.7,
        )

        return fig

    def plotly_cv_roc_auc(self, significance_level=0.05, width=800, height=800):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        # Loop through each fold
        for i in range(self.n_bins):
            y_true = self.fitted[i]["val"][self.target]
            y_pred = self.fitted[i]["val"]["prob"]

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # Create empty figure
        fig = go.Figure()

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.trapz(mean_tpr, mean_fpr)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # Add the shaded region
        fig.add_trace(
            go.Scatter(
                x=np.concatenate((mean_fpr, mean_fpr[::-1])),
                y=np.concatenate((tprs_upper, tprs_lower[::-1])),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Mean ROC Curve +/- 1 SD",
            )
        )

        # Loop through each fold
        for i in range(self.n_bins):
            y_true = self.fitted[i]["val"][self.target]
            y_pred = self.fitted[i]["val"]["prob"]

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            # Add individselfl ROC lines with hover info, no legend
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    hovertemplate=f"Fold {i+1}"
                    + "<br>FPR: %{x:.3f}\
<br>TPR: %{y:.3f}<extra></extra>",
                    legendgroup="Folds",
                    opacity=0.5,
                    line=dict(dash="dot"),
                    showlegend=False,
                )
            )

        # Add mean ROC line
        fig.add_trace(
            go.Scatter(
                x=mean_fpr,
                y=mean_tpr,
                mode="lines+markers",
                name=f"Mean ROC (AUC = {mean_auc:.2f})",
                hovertemplate="Mean FPR: %{x:.3f}<br>Mean TPR: %{y:.3f}\
<extra></extra>",
                line=dict(color="royalblue", width=1),
                marker=dict(color="royalblue", size=5, symbol="circle-open"),
            )
        )

        # Add random guess line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                hovertext="Random Guess",
                line=dict(color="black", dash="dash"),
            )
        )

        # Add titles and labels
        fig.update_layout(
            title=f"ROC Curve (AUC = {mean_auc:.2f})",
            xaxis_title="False Positive Rate (FPR) \
(= 1 - Specificity = FP / (FP + TN))",
            yaxis_title="True Positive Rate (TPR) (= Sensitivity = TP / (TP + FN))",
            width=width,
            height=height,
            legend=dict(
                x=0.65,
                y=0.9,
                # fontsize=16,
                bordercolor="Black",
                borderwidth=1,
            ),
        )

        # Add coefficient estimate and significance statement
        coef, std_err, pvalue = self.fit.params[1], self.fit.se[1], self.fit.pvalues[1]
        significance_statement = self._get_significance_band(pvalue, "coefficient")

        # 95% confidence interval assuming normal distribution
        ci_lower = coef - 1.96 * std_err
        ci_upper = coef + 1.96 * std_err

        # Add annotation inside a box
        fig.add_annotation(
            x=0.75,
            y=0.25,
            text=f"Estimated Coefficient: {coef:.2f}<br>\
95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]<br>\
p-value: {pvalue:.2f}<br>\
{significance_statement}",
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=2,
        )

        fig.show()

    def _compute_auc_variance(self):
        """
        Compute the variance of the AUC estimator. This is used in the
        computation of the DeLong test, and is based on the following paper:
        
        @article{delong1988comparing,
        title={Comparing the areas under two or more correlated receiver operating \
            characteristic curves: a nonparametric approach},
        author={DeLong, Elizabeth R and DeLong, David M and Clarke-Pearson, Daniel L},
        journal={Biometrics},
        pages={837--845},
        year={1988},
        publisher={JSTOR}
        }

        Var(AUC) = (V1 + V2 + V3) / (n1 * n0)

        where
        - V1 = AUC * (1 - AUC)
        - V2 = (n1 - 1) * (Q1 - AUC^2)
        - V3 = (n0 - 1) * (Q0 - AUC^2)
        - Q1 = AUC / (2 - AUC)
        - Q0 = (2 * AUC^2) / (1 + AUC)
        - n1 = number of positive classes
        - n0 = number of negative classes

        Parameters
        ----------
        None. Relies on the following class attributes:
            - self.fitted: dictionary of fitted models and metrics
            - self.target: name of the target variable
            
        Returns
        -------
        var_auc : variance of the AUC estimator
        """
        y_true = self.fit.y
        auc = roc_auc_score(self.fit.y, self.fit.yhat)

        # Count of positive and negative classes
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        # Q1 and Q2 for variance calculation
        Q1 = auc / (2 - auc)
        Q0 = (2 * auc**2) / (1 + auc)

        # Compute the variance
        var_auc = (
            auc * (1 - auc) + (n1 - 1) * (Q1 - auc**2) + (n0 - 1) * (Q0 - auc**2)
        ) / (n1 * n0)

        return var_auc

    def _delong_test_against_chance(self):
        """
        Implement the DeLong test to compare the ROC AUC against the 45-degree
        line (AUC = 0.5).

        The DeLong test uses the Central Limit Theorem (CLT) to approximate
        the distribution of the AUC as normal. The test computes the covariance
        matrix of the paired AUC differences and uses it to generate a Z-statistic.
        According to CLT, this Z-statistic will be approximately normally distributed
        for sufficiently large sample sizes (typically n > 30).

        Parameters
        ----------
        None. Relies on the following class attributes:
            - self.fitted: dictionary of fitted models and metrics
            - self.target: name of the target variable

        Returns:
        z_stat : Z-statistic
        p_value : p-value of the test
        """
        # Get the true and predicted values
        y_true = self.fit.y
        y_pred = self.fit.yhat

        # Calculate the AUC of the model
        auc = roc_auc_score(y_true, y_pred)

        # Compute the variance of the AUC estimator
        var_auc = self._compute_auc_variance()

        # Calculate the Z-statistic against the 45-degree line (AUC=0.5)
        z_stat = (auc - 0.5) / np.sqrt(var_auc)

        # Calculate the p-value
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

        return z_stat[0], p_value[0]

    def plot_cv_roc_auc(self, ax=None, return_all=False, significance_level=0.05):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        # Plotting the ROC Curve - Mean FPR and TPR

        # FPR = False Positive Rate, and is the x-axis
        mean_fpr = np.linspace(0, 1, 100)

        # TPR = True Positive Rate, and is the y-axis
        tprs = []

        # We need to calculate the TPR for each fold at each FPR:
        # 1. Get the true and predicted values for each fold
        # 2. Calculate the FPR and TPR for each fold
        # 3. Interpolate the TPR for each fold at the mean FPR
        # 4. Calculate the mean TPR across all folds

        # Loop through each fold
        for i in range(self.n_bins):
            # Get the true and predicted values for the current fold
            y_true = self.fitted[i]["val"][self.target]
            y_pred = self.fitted[i]["val"]["prob"]

            # Calculate the FPR and TPR for the current fold using the
            # sklearn roc_curve function
            fpr, tpr, _ = roc_curve(y_true, y_pred)

            # Interpolate the TPR for the current fold at the mean FPR
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            # Plot the ROC curve for the current fold
            RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
            ).plot(
                ax=ax,
                # to ensure that the legend is not repeated
                # for each fold
                # label=None,
                alpha=0.4,
            )

        # Calculate the mean TPR across all folds
        mean_tpr = np.mean(tprs, axis=0)

        # Calculate the mean AUC across all folds, using the trapezoidal rule
        # to approximate the integral of the ROC curve
        mean_auc = np.trapz(mean_tpr, mean_fpr)
        self.mean_auc = mean_auc
        self.sd_auc = np.std(
            [
                roc_auc_score(
                    self.fitted[i]["val"][self.target], self.fitted[i]["val"]["prob"]
                )
                for i in range(self.n_bins)
            ]
        )

        # Calculate the standard deviation of the TPR across all folds
        std_tpr = np.std(tprs, axis=0)

        # Our SD band will be mean_tpr +/- 1 std_tpr
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # Plotting the ROC Curve - ax.fill_between gives us the SD band
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2)

        # Set the border on the SD band to be darker and thicker
        ax.spines["bottom"].set_color("grey")
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

        # Plotting the ROC Curve - ax.plot plots the mean ROC curve
        # ax.plot(mean_fpr, mean_tpr, "b--")
        ax.plot(mean_fpr, mean_tpr, "b--", label=f"Mean ROC (AUC = {mean_auc:.2f})")

        # Plotting the random guess line to compare against
        # ax.plot([0, 1], [0, 1], "k--")
        ax.plot([0, 1], [0, 1], "k--", label="Random Guess")

        # Adding a blank bar to the legend to show the SD band
        # ax.plot([], [], " ")
        ax.plot([], [], " ", label="Grey band = +/- 1 SD")

        # Set x and y labels
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        # Perform the DeLong test against the 45-degree line (AUC=0.5)
        dl_stat, p_value = self._delong_test_against_chance()

        # Annotating the plot with hypothesis testing info
        significance_message = f"DeLong Test Statistic\nAgainst the\n45-degree Line=\
\n{dl_stat:.3f}\n\n\
p-value = {p_value:.2f}"
        subtitle_message = (
            f"\nROC AUC is significantly different from 0.5 at the \
{significance_level:.0%} level"
            if p_value < significance_level
            else f"\nROC AUC is not significantly different from 0.5 at the \
{significance_level:.0%} level"
        )

        ax.annotate(
            significance_message,
            xy=(0.7, 0.43),
            xycoords="axes fraction",
            fontsize=16,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="aliceblue",
                alpha=0.5,
            ),
        )

        ax.set_title(f"ROC Curve (AUC = {mean_auc:.2f}){subtitle_message}")

        # Add coefficient estimate and significance statement
        coef, std_err, pvalue = self.fit.params[1], self.fit.se[1], self.fit.pvalues[1]
        significance_statement = self._get_significance_band(pvalue, "coefficient")

        # 95% confidence interval assuming normal distribution
        ci_lower = coef - 1.96 * std_err
        ci_upper = coef + 1.96 * std_err

        text = f"""Logistic Regression Model Fit
-----------------------------
Estimated Coefficient: {coef:.2f}
95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]
p-value: {pvalue:.2f}
{significance_statement}"""

        # Add annotation inside a box
        ax.annotate(
            text,
            xy=(0.36, 0.19),
            xycoords="axes fraction",
            fontsize=16,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="white",
                alpha=0.5,
            ),
        )

        # Create custom legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="b",
                lw=2,
                linestyle="--",
                label=f"Mean ROC (AUC = {mean_auc:.2f})",
            ),
            Line2D([0], [0], color="k", lw=2, linestyle="--", label="Random Guess"),
            Patch(
                facecolor="grey",
                edgecolor="grey",
                alpha=0.2,
                label="Mean(ROC) +/- 1 SD(ROC)",
            ),
        ]

        # ax.legend(loc="lower right", fontsize=16)
        ax.legend(handles=legend_elements, loc="lower right")

        ax = rotate_x_lab(ax)
        ax.figure.tight_layout()
        if return_all:
            return (
                ax,
                mean_fpr,
                tprs,
                tprs_lower,
                tprs_upper,
                mean_auc,
                mean_tpr,
                std_tpr,
            )
        else:
            return ax

    def plot_quintile_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        feature = self.GetVal()[self.feature]
        if self.type == "categorical":
            feature = feature.astype("category")
        observed_target = self.val[self.target]
        predicted_target = self.fit.model.predict(self.val[[self.feature]])
        ax = _quintile_lift_plot(feature, observed_target, predicted_target, ax=ax)
        return ax

    def plotly_quintile_lift_plot(
        feature: pd.Series,
        observed_target: pd.Series,
        modeled_target: pd.Series,
        modeled_color: str = "red",
        observed_color: str = "lightgreen",
    ):
        # Create DataFrame to hold all the data
        df = pd.DataFrame(
            {
                "feature": feature,
                "observed_target": observed_target,
                "modeled_target": modeled_target,
            }
        )

        # Create quintile bins based on the modeled target
        # If there are n < 5 unique values, don't bin quintiles -- instead, bin into
        # n bins based on the modeled target
        unique_values = df["modeled_target"].unique()
        quintile_bins = len(unique_values) if len(unique_values) < 5 else 5
        df["quintile"] = (
            pd.qcut(
                df["modeled_target"], quintile_bins, labels=False, duplicates="drop"
            )
            + 1
        )

        # Calculate the mean target and modeled target for each quintile
        lift_df = (
            df.groupby("quintile")
            .agg(
                observed_target_mean=("observed_target", "mean"),
                modeled_target_mean=("modeled_target", "mean"),
            )
            .reset_index()
        )

        # Create the figure for the quintile lift plot
        fig = go.Figure()

        # Add observed target bars
        fig.add_trace(
            go.Bar(
                x=lift_df["quintile"],
                y=lift_df["observed_target_mean"],
                name="Observed",
                marker_color=observed_color,
            )
        )

        # Add modeled target bars
        fig.add_trace(
            go.Bar(
                x=lift_df["quintile"],
                y=lift_df["modeled_target_mean"],
                name="Modeled",
                marker_color=modeled_color,
            )
        )

        # KL Divergence calculation
        kl_div = kl_divergence(
            lift_df["observed_target_mean"].values,
            lift_df["modeled_target_mean"].values,
        )

        # Gini calculation
        gini_coeff = gini_coefficient(df["observed_target"], df["modeled_target"])

        # Add KL divergence and Gini coefficient as annotations
        fig.add_annotation(
            text=f"KL Divergence: {kl_div:.3f}<br>Gini Coefficient: {gini_coeff:.3f}",
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.05,
            showarrow=False,
            align="right",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )

        # Update layout
        fig.update_layout(
            title="Quintile Lift Plot",
            xaxis_title="Modeled Quintile",
            yaxis_title="Mean Target",
            barmode="group",
            legend_title_text="Legend",
        )

        return fig

    def plot_cat_lift_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        feature = self.GetVal()[self.feature]
        if self.type == "categorical":
            feature = feature.astype("category")
        observed_target = self.val[self.target]
        df = pd.DataFrame(
            {f"{self.feature}": feature, f"{self.target}": observed_target}
        )
        ax = _plot_lift_chart(df=df, feature=self.feature, target=self.target, ax=ax)

        ax.set_title(
            f"Lift Plot - Model Including [{self._plot_label(self.feature)}] \
vs Null Model"
        )
        return ax

    def plotly_cat_lift_plot(self, alpha=0.5):
        """
        Plots the lift chart for a given categorical feature and target.
        """
        feature = self.feature
        target = self.target

        # Create a copy of the data
        df = self.train[[feature, target]].copy()

        # Calculate overall positive rate
        overall_positive_rate = df[target].mean()

        # Group by the feature and calculate the mean target variable
        lift_data = df.groupby(feature)[target].mean().reset_index()
        lift_data["lift"] = lift_data[target] / overall_positive_rate

        # Sort by the lift
        lift_data = lift_data.sort_values("lift", ascending=False)

        # Define the colors based on lift value
        colors = ["green" if lift > 1 else "red" for lift in lift_data["lift"]]

        # Create the figure for the lift chart
        fig = go.Figure()

        # Add the lift bars
        fig.add_trace(
            go.Bar(
                x=lift_data[feature],
                y=lift_data["lift"],
                marker_color=colors,
                opacity=alpha,
            )
        )

        # Add line at lift=1
        fig.add_hline(
            y=1, line=dict(color="black", dash="dash"), annotation_text="Baseline"
        )

        # Add annotations for lift values
        annotations = []
        for i, row in lift_data.iterrows():
            annotations.append(
                dict(
                    x=row[feature],
                    y=row["lift"],
                    text=f"{row['lift']:.2f}",
                    font=dict(family="Arial", size=16, color="black"),
                    showarrow=False,
                )
            )

        # Update layout
        fig.update_layout(
            title_text="Lift Chart for " + feature,
            xaxis_title=feature,
            yaxis_title="Lift",
            showlegend=False,
            annotations=annotations,
        )

        return fig

    def plot_density(self, ax=None, cv_alpha=0.2):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        data = self.GetTrain()[[self.feature, self.target]]

        if self.type == "categorical":
            data[self.feature] = data[self.feature].astype("category")

        # Plot the violin plot
        outlier_df = self.train[[self.feature, self.target]]

        ax = _plot_violin(
            target=data[self.target],
            feature=data[self.feature],
            outlier_df=outlier_df,
            cv_folds_data=self.cv_idx,
            cv_alpha=cv_alpha,
            ax=ax,
            dropvals=[-0.01, -1],
        )

        return ax

    def plotly_density(self, cv_alpha=0.2):
        data = self.GetTrain()[[self.feature, self.target]]

        if self.type == "categorical":
            data[self.feature] = data[self.feature].astype("category")

        # Prepare the data
        data[self.feature] = data[self.feature].cat.remove_unused_categories()
        categories = data[self.feature].cat.categories

        # Initialize figure
        fig = go.Figure()

        # Plot the violin plot
        for category in categories:
            category_data = data[data[self.feature] == category]
            fig.add_trace(
                go.Violin(
                    y=category_data[self.target],
                    name=str(category),
                    box_visible=True,
                    meanline_visible=True,
                    opacity=cv_alpha,
                    points="outliers",  # or 'all' or False
                )
            )

        # Customize layout
        fig.update_layout(
            title=f"Lift Plot - Model Including [{self._plot_label(self.feature)}] vs Null Model",
            yaxis_title=self._plot_label(self.target),
            xaxis_title=self._plot_label(self.feature),
        )

        return fig

    def categorical_plots(self, return_all=False):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.figsize, layout="constrained"
        )
        # Plotting the Bar Chart
        ax1 = self.plot_cv_roc_auc(ax=ax1)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plotting the Stacked Bar Chart
        ax2 = self.plot_stacked_bar_chart(ax=ax2)

        # Plotting the Chi-Sqselfred Test
        ax3 = self.plot_cat_lift_plot(ax=ax3)

        # Plotting the Point Plot
        ax4 = self.plot_point_plot(ax=ax4)

        title_text = f"Small Buisiness Univariate Analysis - {self.feature}\n"
        fig.suptitle(
            title_text,
            fontweight="bold",
            #  fontsize=20
        )
        plt.tight_layout()

        if return_all:
            return fig, ((ax1, ax2), (ax3, ax4))
        else:
            plt.show()

    def continuous_plots(self, return_all=False):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.figsize, layout="constrained"
        )
        # Plotting the ROC Curve
        ax1 = self.plot_cv_roc_auc(ax=ax1)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plotting the Density Plot
        ax2 = self.plot_density(ax=ax2)

        # Plotting the quintile lift plot
        ax3 = self.plot_quintile_plot(ax=ax3)

        # Plotting the empirical CDF
        ax4 = self.plot_cdf(ax=ax4)

        title_text = f"Small Buisiness Univariate Analysis - {self.feature}\n"
        fig.suptitle(
            title_text,
            fontweight="bold",
            #  fontsize=20
        )
        plt.tight_layout()
        if return_all:
            return fig, ((ax1, ax2), (ax3, ax4))
        else:
            plt.show()

    def plot(self, return_all=False, save_all=False, save_path=None):
        if self.type == "categorical":
            if return_all:
                fig, ((ax1, ax2), (ax3, ax4)) = self.categorical_plots(
                    return_all=return_all
                )

                if save_all:
                    fig.savefig(f"{save_path}/{self.feature}.png")
                    plt.close(fig)
                else:
                    return fig, ((ax1, ax2), (ax3, ax4))
            else:
                self.categorical_plots(return_all=return_all)
        elif self.type == "continuous":
            if return_all:
                fig, ((ax1, ax2), (ax3, ax4)) = self.continuous_plots(
                    return_all=return_all
                )

                if save_all:
                    fig.savefig(f"{save_path}/{self.feature}.png")
                    plt.close(fig)
                else:
                    return fig, ((ax1, ax2), (ax3, ax4))
            else:
                self.continuous_plots(return_all=return_all)
        else:
            raise ValueError(f"Type {self.type} is not supported.")

    def calculate_woe_iv(self):
        """
        Calculate Weight of Evidence (WoE) and Information Value (IV) for a given
        feature and target variable.

        Parameters
        ----------
        None. Relies on the following class attributes:
            - self.train: training data
            - self.feature: name of the feature variable
            - self.target: name of the target variable

        Returns
        -------
        tuple: A tuple containing the Information Value (IV) for the feature and a
        Series of Weight of Evidence (WoE) values indexed by feature categories.

        Note
        ----
        - WoE is log(odds of good / odds of bad) for each category in the feature.
        - IV is a metric that qselfntifies the predictive power of the feature, and is
          calculated as sum((%good - %bad) * WoE).
        """
        # Get the feature and target variables from the class attributes
        df = self.train.copy()[[self.feature, self.target]]
        feature = self.feature
        target = self.target

        if self.type == "continuous":
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
            df[feature] = df[feature].fillna(0)
        elif self.type == "categorical":
            if self.is_binary:
                df[feature] = df[feature].fillna(0)
            else:
                dummies = self.train.copy()[[feature]]
                dummies = pd.get_dummies(dummies, prefix=feature).fillna(0)
        else:
            raise ValueError(f"Type {self.type} is not supported.")

        # Calculate the total number of 'good' (target=1) and 'bad' (target=0) in
        # the dataset
        all_good = len(df[df[target] == 1])
        all_bad = len(df[df[target] == 0])

        # Create a cross-tabulation of the feature against the target
        # Normalize by columns to get the proportion of 'good' and 'bad' for each
        # category of the feature
        df_crosstab = pd.crosstab(df[feature], df[target], normalize="columns")

        # Calculate WoE for each category in the feature
        # WoE = log(% of 'good' / % of 'bad')
        df_woe = df_crosstab.assign(WoE=lambda dfx: np.log(dfx[1] / dfx[0]))

        # Calculate IV for each category and sum them up to get the IV for the feature
        # IV = sum((% of 'good' - % of 'bad') * WoE)
        df_iv = df_woe.assign(IV=lambda dfx: (dfx[1] - dfx[0]) * dfx["WoE"])

        # Calculate the total IV for the feature
        feature_iv = df_iv["IV"].sum()

        # Calculate weighted mean of WoE
        n_obs = pd.crosstab(df[feature], df[target]).sum(axis=1)
        weighted_woe = (df_iv["WoE"] * n_obs).sum() / n_obs.sum()

        return feature_iv, weighted_woe

    def _mallows_cp(self, y, yhat, k):
        """
        Calculate the Mallows Cp statistic for a given model.

        Parameters
        ----------
        y : array-like
            True values of the target variable
        yhat : array-like
            Predicted values of the target variable
        k : int
            Number of features in the model

        Returns
        -------
        float: Mallows Cp statistic
        """
        n = len(y)
        mse = mean_squared_error(y, yhat)
        return mse + 2 * k * mse / n

    def _ttest_pvalue(self, y, yhat):
        """
        Calculate the p-value from a t-test comparing the distributions of the
        true and predicted values.

        Parameters
        ----------
        y : array-like
            True values of the target variable
        yhat : array-like
            Predicted values of the target variable

        Returns
        -------
        float: p-value from a t-test comparing the distributions of the true and
        predicted values
        """
        _, p_value = ttest_ind(y, yhat)
        return p_value

    def binary_cv_output(self, threshold=0.5):
        """
        Creates a data frame with the most important statistics for a binary target
        variable (cross-validation statistics):

        1. Name of the feature variable (self.feature)
        2. Fold number (from self.fitted)
        3. Information Value (IV) of the feature for each fold (from
           self.calculate_woe_iv())
        4. Weight of Evidence (WoE) of the feature for each fold(from
           self.calculate_woe_iv())
        5a. Calculated AUC (from auc(self.fitted[i]['val'][self.target],
            self.fitted[i]['val']['prob'])) for each fold
        5b. Calculated SD of AUC (from self._compute_auc_variance()) for each fold
        6a. Fitted coefficient (from self.fit.model.coef_) for each fold
        6b. Fitted intercept (from self.fit.model.intercept_) for each fold
        6c. Fitted coefficient p-value (from self.fit.pvalues_) for each fold
        6d. Fitted intercept p-value (from self.fit.pvalues_) for each fold
        7a. Training Accuracy (from acc(self.fit.y, self.fit.yhat)) for each fold
        7b. Validation Accuracy (from acc(self.val[self.target],
            self.fit.model.predict(self.val))) for each fold
        7c. p-value from a t-test comparing training and validation accuracy
            distributions (only one value here - uses the entire training and
            validation sets)
        8a. Training Precision (from precision_score(self.fit.y, self.fit.yhat)) for
            each fold
        8b. Validation Precision (from precision_score(self.val[self.target],
            self.fit.model.predict(self.val)))
        8c. p-value from a t-test comparing training and validation precision
            distributions (only one value here - uses the entire training and
            validation sets)
        9a. Training Recall (from recall_score(self.fit.y, self.fit.yhat)) for
            each fold
        9b. Validation Recall (from recall_score(self.val[self.target],
            self.fit.model.predict(self.val)))
        9c. p-value from a t-test comparing training and validation recall
            distributions (only one value here - uses the entire training and
            validation sets)
        10a. Training F1 Score (from f1_score(self.fit.y, self.fit.yhat)) for
             each fold
        10b. Validation F1 Score (from f1_score(self.val[self.target],
             self.fit.model.predict(self.val)))
        10c. p-value from a t-test comparing training and validation F1 score
             distributions (only one value here - uses the entire training and
             validation sets)
        11a. Training balanced accuracy score (from balanced_accuracy_score(self.fit.y,
             self.fit.yhat)) for each fold
        11b. Validation balanced accuracy score (from
             balanced_accuracy_score(self.val[self.target],
             self.fit.model.predict(self.val)))
        11c. p-value from a t-test comparing training and validation balanced accuracy
             score distributions (only one value here - uses the entire training and
             validation sets)
        12a. Training negative log loss (from log_loss(self.fit.y, self.fit.yhat)) for
             each fold
        12b. Validation negative log loss (from log_loss(self.val[self.target],
             self.fit.model.predict(self.val)))
        12c. p-value from a t-test comparing training and validation negative log
             loss distributions (only one value here - uses the entire training and
             validation sets)
        13a. Training hinge loss (from hinge_loss(self.fit.y, self.fit.yhat)) for
             each fold
        13b. Validation hinge loss (from hinge_loss(self.val[self.target],
             self.fit.model.predict(self.val))) for each fold
        13c. p-value from a t-test comparing training and validation hinge loss
             distributions (only one value here - uses the entire training and
             validation sets)
        14a. Training Matthews correlation coefficient
             (from matthews_corrcoef(self.fit.y, self.fit.yhat)) for each fold
        14b. Validation Matthews correlation coefficient
             (from matthews_corrcoef(self.val[self.target],
             self.fit.model.predict(self.val))) for each fold
        14c. p-value from a t-test comparing training and validation Matthews
             correlation coefficient distributions (only one value here - uses the
             entire training and validation sets)
        15a. Training Informedness (from informedness(self.fit.y, self.fit.yhat)) for
             each fold
        15b. Validation Informedness (from informedness(self.val[self.target],
             self.fit.model.predict(self.val))) for each fold
        15c. p-value from a t-test comparing training and validation Informedness
             distributions (only one value here - uses the entire training and
             validation sets) for each fold
        16a. Training Markedness (from markedness(self.fit.y, self.fit.yhat)) for
             each fold
        16b. Validation Markedness (from markedness(self.val[self.target],
             self.fit.model.predict(self.val))) for each fold
        16c. p-value from a t-test comparing training and validation Markedness
                distributions (only one value here - uses the entire training and
                validation sets) for each fold
        """
        Y = namedtuple("Y", [f"f{i}" for i in range(self.n_bins)])
        y = Y(
            **{
                f"f{i}": self.fitted[i]["train"][self.target]
                for i in range(self.n_bins)
            }
        )
        yhat = Y(
            **{f"f{i}": self.fitted[i]["train"]["prob"] for i in range(self.n_bins)}
        )

        y_val = Y(
            **{f"f{i}": self.fitted[i]["val"][self.target] for i in range(self.n_bins)}
        )
        yhat_val = Y(
            **{f"f{i}": self.fitted[i]["val"]["prob"] for i in range(self.n_bins)}
        )

        binary = {}
        # 1. Name of the feature variable (self.feature)
        binary["feature"] = [self.feature] * self.n_bins
        # 2. Fold number (from self.fitted)
        binary["fold"] = range(self.n_bins)
        # 3. Information Value (IV) of the feature for each fold (from
        #    self.calculate_woe_iv())
        binary["y_pvalue"] = [
            self._ttest_pvalue(y[i], yhat[i]) for i in range(self.n_bins)
        ]
        binary["iv"] = [self.calculate_woe_iv()[0]] * self.n_bins
        # 4. Weight of Evidence (WoE) of the feature for each fold(from
        #    self.calculate_woe_iv())
        binary["woe"] = [self.calculate_woe_iv()[1]] * self.n_bins
        # 5a. Calculated AUC (from auc(self.fitted[i]['val'][self.target],
        #     self.fitted[i]['val']['prob'])) for each fold
        binary["auc"] = [roc_auc_score(y[i], yhat[i]) for i in range(self.n_bins)]
        # 5b. Calculated SD of AUC (from self._compute_auc_variance()) for each fold
        binary["sd_auc"] = [np.sqrt(self._compute_auc_variance())[0]] * self.n_bins
        # 6a. Fitted coefficient (from self.fit.model.coef_) for each fold
        binary["coef"] = [self.fit.model.params[1]] * self.n_bins
        # 6b. Fitted intercept (from self.fit.model.intercept_) for each fold
        binary["intercept"] = [self.fit.model.params[0]] * self.n_bins
        # 6c. Fitted coefficient p-value (from self.fit.pvalues_) for each fold
        binary["coef_pval"] = [self.fit.pvalues[1]] * self.n_bins
        # 6d. Fitted intercept p-value (from self.fit.pvalues_) for each fold
        binary["intercept_pval"] = [self.fit.pvalues[0]] * self.n_bins
        # 7a. Training Accuracy (from acc(self.fit.y, self.fit.yhat)) for each fold
        binary["train_acc"] = [
            acc(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 7b. Validation Accuracy (from acc(self.val[self.target],
        #     self.fit.model.predict(self.val))) for each fold
        binary["val_acc"] = [
            acc(pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]

        # 8a. Training Precision (from precision_score(self.fit.y, self.fit.yhat)) for
        #     each fold
        binary["train_precision"] = [
            precision_score(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 8b. Validation Precision (from precision_score(self.val[self.target],
        #     self.fit.model.predict(self.val)))
        binary["val_precision"] = [
            precision_score(
                pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int)
            )
            for i in range(self.n_bins)
        ]

        # 9a. Training Recall (from recall_score(self.fit.y, self.fit.yhat)) for
        #     each fold
        binary["train_recall"] = [
            recall_score(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 9b. Validation Recall (from recall_score(self.val[self.target],
        #     self.fit.model.predict(self.val)))
        binary["val_recall"] = [
            recall_score(
                pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int)
            )
            for i in range(self.n_bins)
        ]

        # 10a. Training F1 Score (from f1_score(self.fit.y, self.fit.yhat)) for
        #      each fold
        binary["train_f1"] = [
            f1_score(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 10b. Validation F1 Score (from f1_score(self.val[self.target],
        #      self.fit.model.predict(self.val)))
        binary["val_f1"] = [
            f1_score(pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]

        # 11a. Training balanced accuracy score (from balanced_accuracy_score(self.fit.y,
        #      self.fit.yhat)) for each fold
        binary["train_bal_acc"] = [
            balanced_accuracy_score(
                pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int)
            )
            for i in range(self.n_bins)
        ]
        # 11b. Validation balanced accuracy score (from
        #      balanced_accuracy_score(self.val[self.target],
        #      self.fit.model.predict(self.val)))
        binary["val_bal_acc"] = [
            balanced_accuracy_score(
                pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int)
            )
            for i in range(self.n_bins)
        ]

        # 12a. Training negative log loss (from log_loss(self.fit.y, self.fit.yhat)) for
        #      each fold
        binary["train_neg_log_loss"] = [
            log_loss(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 12b. Validation negative log loss (from log_loss(self.val[self.target],
        #      self.fit.model.predict(self.val)))
        binary["val_neg_log_loss"] = [
            log_loss(pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]

        # 13a. Training hinge loss (from hinge_loss(self.fit.y, self.fit.yhat)) for
        #      each fold
        binary["train_hinge_loss"] = [
            hinge_loss(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 13b. Validation hinge loss (from hinge_loss(self.val[self.target],
        #      self.fit.model.predict(self.val))) for each fold
        binary["val_hinge_loss"] = [
            hinge_loss(pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]

        # 14a. Training Matthews correlation coefficient
        #      (from matthews_corrcoef(self.fit.y, self.fit.yhat)) for each fold
        binary["train_mcc"] = [
            matthews_corrcoef(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 14b. Validation Matthews correlation coefficient
        #      (from matthews_corrcoef(self.val[self.target],
        #      self.fit.model.predict(self.val))) for each fold
        binary["val_mcc"] = [
            matthews_corrcoef(
                pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int)
            )
            for i in range(self.n_bins)
        ]

        # 15a. Training Informedness (from informedness(self.fit.y, self.fit.yhat)) for
        #      each fold
        binary["train_informedness"] = [
            informedness(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 15b. Validation Informedness (from informedness(self.val[self.target],
        #      self.fit.model.predict(self.val))) for each fold
        binary["val_informedness"] = [
            informedness(
                pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int)
            )
            for i in range(self.n_bins)
        ]

        # 16a. Training Markedness (from markedness(self.fit.y, self.fit.yhat)) for
        #      each fold
        binary["train_markedness"] = [
            markedness(pd.Series(y[i]), pd.Series(yhat[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]
        # 16b. Validation Markedness (from markedness(self.val[self.target],
        #      self.fit.model.predict(self.val))) for each fold
        binary["val_markedness"] = [
            markedness(pd.Series(y_val[i]), pd.Series(yhat_val[i]).ge(0.5).astype(int))
            for i in range(self.n_bins)
        ]

        return pd.DataFrame(binary)

    def binary_target_output(self, threshold=0.5):
        """
        Creates a data frame with the most important statistics for a binary target variable:

        1. Number of observations (from self.train.shape[0])
        2. Name of the feature variable (self.feature)
        3. Information Value (IV) of the feature (from self.calculate_woe_iv())
        4. Weight of Evidence (WoE) of the feature (from self.calculate_woe_iv())
        5. Calculated AUC (from auc(self.fit.y, self.fit.yhat))
        6. Calculated SD of AUC (from self._compute_auc_variance())
        6a. Fitted coefficient (from self.fit.model.coef_) for each fold
        6b. Fitted intercept (from self.fit.model.intercept_) for each fold
        6c. Fitted coefficient p-value (from self.fit.pvalues_) for each fold
        6d. Fitted intercept p-value (from self.fit.pvalues_) for each fold
        7a. Training Accuracy (from acc(self.fit.y, self.fit.yhat)) for each fold
        7b. Validation Accuracy (from acc(self.val[self.target], self.fit.model.predict(self.val))) for each fold
        8a. Training Precision (from precision_score(self.fit.y, self.fit.yhat))
        8b. Validation Precision (from precision_score(self.val[self.target], self.fit.model.predict(self.val)))
        9a. Training Recall (from recall_score(self.fit.y, self.fit.yhat))
        9b. Validation Recall (from recall_score(self.val[self.target], self.fit.model.predict(self.val)))
        10a. Training F1 Score (from f1_score(self.fit.y, self.fit.yhat))
        10b. Validation F1 Score (from f1_score(self.val[self.target], self.fit.model.predict(self.val)))
        11a. Training balanced accuracy score (from balanced_accuracy_score(self.fit.y, self.fit.yhat))
        11b. Validation balanced accuracy score (from balanced_accuracy_score(self.val[self.target], self.fit.model.predict(self.val)))
        12a. Training negative log loss (from log_loss(self.fit.y, self.fit.yhat))
        12b. Validation negative log loss (from log_loss(self.val[self.target], self.fit.model.predict(self.val)))
        13a. Training hinge loss (from hinge_loss(self.fit.y, self.fit.yhat))
        13b. Validation hinge loss (from hinge_loss(self.val[self.target], self.fit.model.predict(self.val)))
        14a. Training Matthews correlation coefficient (from matthews_corrcoef(self.fit.y, self.fit.yhat))
        14b. Validation Matthews correlation coefficient (from matthews_corrcoef(self.val[self.target], self.fit.model.predict(self.val)))
        15a. Training Informedness (from informedness(self.fit.y, self.fit.yhat))
        15b. Validation Informedness (from informedness(self.val[self.target], self.fit.model.predict(self.val)))
        """
        y = self.fit.y
        yhat = self.fit.yhat

        y_val = self.val[self.target]
        yhat_val = self.fit.model.predict(self.val[self.feature])

        binary = {}
        binary["n_obs"] = [self.train.shape[0]]
        binary["feature"] = [self.feature]
        binary["column_type"] = [self.type]
        binary["iv"] = [self.calculate_woe_iv()[0]]
        binary["woe"] = [self.calculate_woe_iv()[1]]
        binary["coef"] = [self.fit.model.params[1]]
        binary["coef_p_value"] = [self.fit.pvalues[1]]
        binary["training_matthews_corrcoef"] = [
            matthews_corrcoef(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_matthews_corrcoef"] = [
            matthews_corrcoef(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_informedness"] = [
            informedness(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_informedness"] = [
            informedness(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_markedness"] = [
            markedness(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_markedness"] = [
            markedness(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_accuracy"] = [
            acc(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_accuracy"] = [
            acc(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_f1_score"] = [
            f1_score(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_f1_score"] = [
            f1_score(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["auc"] = [roc_auc_score(y, yhat)]
        binary["sd_auc"] = [np.sqrt(self._compute_auc_variance())[0]]
        binary["intercept"] = [self.fit.model.params[0]]
        binary["intercept_p_value"] = [self.fit.pvalues[0]]
        binary["training_precision"] = [
            precision_score(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_precision"] = [
            precision_score(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_recall"] = [
            recall_score(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_recall"] = [
            recall_score(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_balanced_accuracy"] = [
            balanced_accuracy_score(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_balanced_accuracy"] = [
            balanced_accuracy_score(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_neg_log_loss"] = [
            log_loss(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_neg_log_loss"] = [
            log_loss(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]
        binary["training_hinge_loss"] = [
            hinge_loss(self.fit.y, self.fit.yhat.gt(threshold).astype(int))
        ]
        binary["validation_hinge_loss"] = [
            hinge_loss(
                self.val[self.target],
                self.fit.model.predict(self.val).gt(threshold).astype(int),
            )
        ]

        return pd.DataFrame(binary)
