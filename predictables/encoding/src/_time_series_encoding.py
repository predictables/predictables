"""Fit a time series model to the lagged mean-encoded categorical variable, and predict the current value of the mean-encoded variable.

Takes a dataframe with the lagged mean-encoded values, an id column, and the
target, and fits a time series model to the lagged mean-encoded values. The
model is then used to predict the current value of the mean-encoded variable.

Each observation in the dataframe is a time series of the mean-encoded
values of a categorical variable, with 18 observations per time series.
We use time-series cross-validation to fit the model using the most recent
12 observations to predict the 13th observation. The model is validated this
way 6 times, using the most recent 12 observations to predict the 13th, 14th,
15th, 16th, 17th, and 18th observations.

The 12 observations are a sliding window, so the first validation set uses
observations 1-12 to predict 13, the second validation set uses observations
2-13 to predict 14, and so on.
"""

import polars as pl
import lightgbm as lgb

from predictables.encoding.src._log_transform import log_transform
from predictables.encoding.src._logit_transform import logit_transform
