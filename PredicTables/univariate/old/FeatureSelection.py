import pandas as pd
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class FeatureSelection:

    def __init__(self,
                df:pd.DataFrame,
                target:str,
                task:str='classification',
                verbose:bool=False):
        self.df = df
        self.target = target
        self.task = task
        self.verbose = verbose

        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X,
                             self.y,
                             test_size=0.2,
                             shuffle=True,
                             random_state=42)

        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_train,
                             self.y_train,
                             test_size=0.25,
                             shuffle=True,
                             random_state=42)

        # hyperparameter grid for logistic regression
        self.logistic_grid = {
            'C': np.logspace(-4, 4, 20),
            'max_iter': [10000],
            'class_weight': [None, 'balanced'],
            'intercept_scaling': [1, 2, 3]
        }
        self.logistic_best_params = None

        self.logistic_model = LogisticRegression(penalty='l1',
                                                n_jobs=-1,
                                                fit_intercept=True,
                                                solver='saga',
                                                tol=0.00001,
                                                warm_start=True,
                                                random_state=42)

        # hyperparameter grid for random forest
        self.rf_grid = {
            'n_estimators': [100, 200, 500, 1000],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 20, 50, 100],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 3],
            'max_features': ['sqrt', 'log2'],
            'max_leaf_nodes': [5, 10, 20, 50, 100],
            'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3],
            'bootstrap': [True],
            'warm_start': [True]
        }
        self.rf_best_params = None

        # hyperparameter grid for xgboost
        self.xgb_grid = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 5, 10, 20, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': [0.5, 0.75, 1.0],
            'colsample_bytree': [0.5, 0.75, 1.0],
            'colsample_bylevel': [0.5, 0.75, 1.0],
            'colsample_bynode': [0.5, 0.75, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            'gamma': [0.0, 0.1, 0.5, 1.0],
            'min_child_weight': [0, 1, 5, 10],
            'max_delta_step': [0, 1, 5, 10],
            'scale_pos_weight': [1, 2, 5, 10],
            'base_score': [0.5, 0.75, 1.0]
        }
        self.xgb_best_params = None

        # hyperparameter grid for lightgbm
        self.lgbm_grid = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 5, 10, 20, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': [0.5, 0.75, 1.0],
            'colsample_bytree': [0.5, 0.75, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            'min_child_weight': [0, 1, 5, 10],
            'min_child_samples': [1, 5, 10, 20],
            'min_split_gain': [0.0, 0.1, 0.5, 1.0],
            'scale_pos_weight': [1, 2, 5, 10]
        }
        self.lgbm_best_params = None

        # hyperparameter grid for catboost
        self.cat_grid = {
            'iterations': [100, 200, 500, 1000],
            'depth': [None, 5, 10, 20, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1],
            'l2_leaf_reg': [0.0, 0.1, 0.5, 1.0],
            'border_count': [1, 5, 10, 20],
            'bagging_temperature': [0.0, 0.1, 0.5, 1.0],
            'random_strength': [0.0, 0.1, 0.5, 1.0],
            'scale_pos_weight': [1, 2, 5, 10]
        }
        self.cat_best_params = None


    def _prep_data(self, X=None):
        """
        Prepares the data for feature selection by one-hot encoding categorical
        features and scaling numerical features.

        Parameters
        ----------
        X : pandas DataFrame
            Dataframe containing the features to be used for feature selection.
            Defaults to None, in which case self.X_train is used.

        Returns
        -------
        X : pandas DataFrame
            Dataframe containing the features to be used for feature selection.
        """
        # if no dataframe is specified, use the training data
        if X is None:
            X = self.X_train

        # get categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # get numerical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

        # one-hot encode the categorical features
        X_train_cat = pd.get_dummies(X[categorical_features], drop_first=True) \
            if len(categorical_features) > 0 else None
        
        # scale the numerical features
        X_train_num = X[numerical_features]
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_num),
                                        columns=numerical_features,
                                        index=X_train_num.index)
        
        # combine the categorical and numerical features with the unadjusted features
        drop_cols = list(set(categorical_features) | set(numerical_features))
        X_other = X.drop(columns=drop_cols)
        X_other = X_other
        dflist = [X_train_scaled]
        if X_train_cat is not None:
            dflist.append(X_train_cat)
        dflist.append(X_other)
        X_train = pd.concat(dflist, axis=1)

        # drop datetime features
        datetime_cols = X_train.select_dtypes(include=['datetime64']).columns.tolist()
        X_train = X_train.drop(columns=datetime_cols)

        # convert boolean features to integers
        bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
        # print(X_train[bool_cols].dtypes)
        # for col in bool_cols:
            # print(X_train[col].value_counts()) 
        X_train.loc[:, bool_cols] = X_train.copy().loc[:, bool_cols].astype(int)

        return X_train

    def _logistic_fine_tune(self):
        """
        Performs a random search on the logistic regression model to fine tune 
        all parameters except for the regularization parameter, which will be used
        as a criterion for feature selection, so needs to be manually specified.
        """
        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train        

        # fit the random search
        logistic_search = GridSearchCV(
            LogisticRegression(penalty='l1',
                               n_jobs=-1,
                               fit_intercept=True,
                               solver='saga',
                               tol=0.0001),
            self.logistic_grid,
            verbose=1)
        logistic_search.fit(X, y)

        # get the best parameters from the random search
        self.logistic_best_params = logistic_search.best_params_
        
    def _fit_single_lasso(self, lambda_):
        """
        Fits a single lasso model with a given L1 regularization parameter. Returns
        the features in the model that were dropped.

        Parameters
        ----------
        lambda_ : float
            L1 regularization parameter.

        Returns
        -------
        dropped_features : list
            List of features that were dropped by the logistic model.
        """
        # Use the initialized logistic model
        self.logistic_model.set_params(C=lambda_, **self.logistic_best_params)
        self.logistic_model.fit(self.X_train, self.y_train)

        # get the features and the coefficients of the logistic model
        features = self.X_train.columns
        coefs = self.logistic_model.coef_[0]

        # get the features that were dropped by the logistic model
        dropped_features = [features[i] for i in range(len(features)) if coefs[i] == 0]

        return dropped_features
        # # perform the random parameter search if it hasn't been done already
        # if self.logistic_best_params is None:
        #     self._logistic_fine_tune()

        # # fit lasso model with given `lambda_` and the best parameters from
        # # the random search
        # lasso_model = LogisticRegression(penalty='l1',
        #                                 n_jobs=-1,
        #                                 fit_intercept=True,
        #                                 tol=0.0001,
        #                                 random_state=42,
        #                                  C=lambda_,
        #                                  **self.logistic_best_params)
        # lasso_model.fit(self.X_train, self.y_train)

        # # get the features and the coefficients of the lasso model
        # features = self.X_train.columns
        # coefs = lasso_model.coef_[0]

        # # get the features that were dropped by the lasso model
        # dropped_features = [features[i] for i in range(len(features))
        # if coefs[i] == 0]

        # return dropped_features

    def worker_lambda_range(self, chunk):
        """
        This function is used by the `lasso_feature_selection` method to fit
        lasso models in parallel. It takes a list of L1 regularization parameters
        and returns a dictionary with the features that were dropped by each model.
        """
        dropped_for_chunk = {}
        for lambda_ in chunk:
            dropped = self._fit_single_lasso(lambda_)
            dropped_for_chunk[lambda_] = dropped
        return dropped_for_chunk
    
    def lasso_path_feature_selection(self, lambda_range=None, save=None):
        # Prepare the data 
        X = self._prep_data(self.X_train)
        y = self.y_train

        # If lambda range is provided, convert it to alpha values for lasso_path.
        # In Lasso, alpha is directly proportional to lambda.
        if lambda_range is not None:
            alphas = 1 / lambda_range
        else:
            alphas = np.logspace(-6, 6, 1000)

        # Compute the Lasso path
        _, coefs, _ = lasso_path(X, y, alphas=alphas)

        # Create a DataFrame to store the smallest alpha at which each feature
        # is eliminated
        alpha_values = pd.DataFrame(np.inf, index=X.columns, columns=['Alpha'])

        for feature_idx, feature_name in enumerate(X.columns):
            # Find the alpha where the feature's coefficient becomes zero
            zero_idx = np.where(coefs[feature_idx, :] == 0)[0]
            if zero_idx.size:
                alpha_values.loc[feature_name] = alphas[zero_idx[0]]

        # Convert alpha values back to lambda for consistency with the rest of your code
        lambda_values = 1 / alpha_values
        lambda_values.columns = ['Lambda']

        # Save the results if required
        if save is not None:
            if save[-4:] != '.csv':
                save += '.csv'
            lambda_values.to_csv(save)

        return lambda_values

    def lasso_feature_selection(self, lambda_range=None, save=None):
        """
        Performs feature selection using lasso regression. The regularization parameter
        ranges over the `lambda_range` parameter, and the method returns a dataframe,
        with, for each feature, the smallest value of lambda that caused that feature to
        be dropped.

        Parameters
        ----------
        lambda_range : list
            List of L1 regularization parameters to use. Defaults to None, in which
            case a default range is used.
        save : str
            Path to save the lambda values to. Defaults to None, in which case the
            lambda values are not saved.
        """
        # keep track of the remaining variables (that haven't been dropped yet)
        remaining_features = self.X_train.columns.to_series()

        # if no lambda range is specified, use a default range
        if lambda_range is None:
            lambda_range = np.logspace(np.log(1e-9), np.log(1000), 10000)

        # Split the lambda_range into chunks for parallel processing
        n_cores = cpu_count()
        chunks = np.array_split(lambda_range, n_cores)

        with Pool(n_cores) as pool:
            results = pool.map(lambda chunk: self.worker_lambda_range(chunk), chunks)

        # Now, combine the results from all worker processes
        all_dropped_features = {}
        for res in results:
            all_dropped_features.update(res)

        # Using all_dropped_features, update the lambda_values
        lambda_values = pd.DataFrame([-1] * self.X_train.shape[1],
                                        columns=self.X_train.columns)
        for lambda_, features in all_dropped_features.items():
            lambda_values.loc[features, :] = lambda_

        # save the lambda values if a path is specified
        if save is not None:
            if save[-4:] != '.csv':
                save += '.csv'
            lambda_values.to_csv(save)

        return lambda_values
        # # if no lambda range is specified, use a default range
        # if lambda_range is None:
        #     lambda_range = np.logspace(0, 1000, 10000)

        # # keep track of the lambda values that caused each feature to be dropped
        # lambda_values = pd.DataFrame([-1] * self.X_train.shape[1],
        #                                 columns=self.X_train.columns)

        # # loop until all features have been dropped from the model
        # for lambda_ in tqdm(lambda_range, desc='Running lasso feature selection...'):
        #     # fit a lasso model with the given lambda
        #     dropped_features = self._fit_single_lasso(lambda_)

        #     # if no features were dropped, but there are still features left,
        #     # then move on to the next lambda value
        #     if len(dropped_features) == 0 and len(remaining_features) > 0:
        #         continue

        #     # if features were dropped, then update the remaining features
        #     # and the lambda values
        #     remaining_features = remaining_features.drop(dropped_features)
        #     lambda_values.loc[dropped_features, :] = lambda_

        #     # if there are no more features left, then break out of the loop
        #     if len(remaining_features) == 0:
        #         break

        # # save the lambda values if a path is specified
        # if save is not None:
        #     if save[-4:] != '.csv':
        #         save += '.csv'
        #     lambda_values.to_csv(save)

        # return lambda_values

    def _rf_fine_tune(self):
        """
        Fine-tunes the random forest model using a grid search. The best parameters
        are saved in the `rf_best_params` attribute.
        """
        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the random search
        rf_search = GridSearchCV(RandomForestClassifier(),self.rf_grid,n_jobs=-1,verbose=1)
        rf_search.fit(X, y)

        # get the best parameters from the random search
        self.rf_best_params = rf_search.best_params_

    def rf_feature_selection(self, save=None):
        """
        Performs feature selection using random forests. Returns a dataframe
        with, for each feature, the number of times that feature was used in
        the random forest model.

        Parameters
        ----------
        save : str
            Path to save the feature importances to. Defaults to None, in which
            case the feature importances are not saved.
        """
        # fine tune the random forest model if it hasn't been done already
        if self.rf_best_params is None:
            self._rf_fine_tune()

        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the random forest model
        rf_model = RandomForestClassifier(random_state=42, **self.rf_best_params)
                                          
        rf_model.fit(X, y)

        # get the features and the feature importances
        features = X.columns
        importances = rf_model.feature_importances_

        # create a dataframe with the feature importances
        rf_feature_importances = pd.DataFrame(importances,
                                                columns=['importance'],
                                                index=features)
        rf_feature_importances = rf_feature_importances.sort_values(by='importance', ascending=False)

        # save the feature importances if a path is specified
        if save is not None:
            if save[-4:] != '.csv':
                save += '.csv'
            rf_feature_importances.to_csv(save)

        return rf_feature_importances

    def _xgb_fine_tune(self):
        """
        Fine-tunes the XGBoost model using a grid search. The best parameters
        are saved in the `xgb_best_params` attribute.
        """
        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the random search
        xgb_search = GridSearchCV(XGBClassifier(),self.xgb_grid,n_jobs=-1,verbose=1)
        xgb_search.fit(X, y)

        # get the best parameters from the random search
        self.xgb_best_params = xgb_search.best_params_

    def xgb_feature_selection(self, save=None):
        """
        Performs feature selection using XGBoost. Returns a dataframe
        with, for each feature, the number of times that feature was used in
        the XGBoost model.

        Parameters
        ----------
        save : str
            Path to save the feature importances to. Defaults to None, in which
            case the feature importances are not saved.
        """
        # fine tune the XGBoost model if it hasn't been done already
        if self.xgb_best_params is None:
            self._xgb_fine_tune()

        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the XGBoost model
        xgb_model = XGBClassifier(random_state=42, **self.xgb_best_params)
                                          
        xgb_model.fit(X, y)

        # get the features and the feature importances
        features = X.columns
        importances = xgb_model.feature_importances_

        # create a dataframe with the feature importances
        xgb_feature_importances = pd.DataFrame(importances,
                                                columns=['importance'],
                                                index=features)
        xgb_feature_importances = xgb_feature_importances.sort_values(by='importance', ascending=False)

        # save the feature importances if a path is specified
        if save is not None:
            if save[-4:] != '.csv':
                save += '.csv'
            xgb_feature_importances.to_csv(save)

        return xgb_feature_importances

    def _lgbm_fine_tune(self):
        """
        Fine-tunes the LightGBM model using a grid search. The best parameters
        are saved in the `lgbm_best_params` attribute.
        """
        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the random search
        lgbm_search = GridSearchCV(LGBMClassifier(),self.lgbm_grid,n_jobs=-1,verbose=1)
        lgbm_search.fit(X, y)

        # get the best parameters from the random search
        self.lgbm_best_params = lgbm_search.best_params_

    def lgbm_feature_selection(self, save=None):
        """
        Performs feature selection using LightGBM. Returns a dataframe
        with, for each feature, the number of times that feature was used in
        the LightGBM model.

        Parameters
        ----------
        save : str
            Path to save the feature importances to. Defaults to None, in which
            case the feature importances are not saved.
        """
        # fine tune the LightGBM model if it hasn't been done already
        if self.lgbm_best_params is None:
            self._lgbm_fine_tune()

        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the LightGBM model
        lgbm_model = LGBMClassifier(random_state=42, **self.lgbm_best_params)
                                          
        lgbm_model.fit(X, y)

        # get the features and the feature importances
        features = X.columns
        importances = lgbm_model.feature_importances_

        # create a dataframe with the feature importances
        lgbm_feature_importances = pd.DataFrame(importances,
                                                columns=['importance'],
                                                index=features)
        lgbm_feature_importances = lgbm_feature_importances.sort_values(by='importance',
                                                                        ascending=False)

        # save the feature importances if a path is specified
        if save is not None:
            if save[-4:] != '.csv':
                save += '.csv'
            lgbm_feature_importances.to_csv(save)

        return lgbm_feature_importances

    def _cat_fine_tune(self):
        """
        Fine-tunes the CatBoost model using a grid search. The best parameters
        are saved in the `cat_best_params` attribute.
        """
        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the random search
        cat_search = GridSearchCV(CatBoostClassifier(),self.cat_grid,n_jobs=-1,verbose=1)
        cat_search.fit(X, y)

        # get the best parameters from the random search
        self.cat_best_params = cat_search.best_params_

    def cat_feature_selection(self, save=None):
        """
        Performs feature selection using CatBoost. Returns a dataframe
        with, for each feature, the number of times that feature was used in
        the CatBoost model.

        Parameters
        ----------
        save : str
            Path to save the feature importances to. Defaults to None, in which
            case the feature importances are not saved.
        """
        # fine tune the CatBoost model if it hasn't been done already
        if self.cat_best_params is None:
            self._cat_fine_tune()

        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the CatBoost model
        cat_model = CatBoostClassifier(random_state=42, **self.cat_best_params)
                                          
        cat_model.fit(X, y)

        # get the features and the feature importances
        features = X.columns
        importances = cat_model.feature_importances_

        # create a dataframe with the feature importances
        cat_feature_importances = pd.DataFrame(importances,
                                                columns=['importance'],
                                                index=features)
        cat_feature_importances = cat_feature_importances.sort_values(by='importance',
                                                                        ascending=False)

        # save the feature importances if a path is specified
        if save is not None:
            if save[-4:] != '.csv':
                save += '.csv'
            cat_feature_importances.to_csv(save)

        return cat_feature_importances

# SelectKBest, chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
    def select_k_best(self, score_func, k=10):
        """
        Performs feature selection using SelectKBest. Returns a dataframe
        with, for each feature, the score that feature received.

        Parameters
        ----------
        score_func : callable
            Function that returns a score for each feature.
        k : int
            Number of features to select.
        """
        # prepare the data for feature selection
        X = self._prep_data(self.X_train)
        y = self.y_train

        # fit the SelectKBest model
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        # get the features and the scores
        features = X.columns
        scores = selector.scores_

        # create a dataframe with the scores
        scores_df = pd.DataFrame(scores,
                                columns=['score'],
                                index=features)
        scores_df = scores_df.sort_values(by='score', ascending=False)

        return scores_df