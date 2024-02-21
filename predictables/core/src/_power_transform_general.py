# import pandas as pd
# import numpy as np
# import polars as pl
# from typing import Union
# from sklearn.preprocessing import PowerTransformer

# from predictables.util import to_pd_s


# def power_transform(x: Union[pd.Series, pl.Series, np.ndarray], transform:str='box-cox') -> Union[pd.Series, pl.Series]:
#     """
#     Apply power transform to the input data

#     Parameters
#     ----------
#     x : Union[pd.Series, pl.Series, np.ndarray]
#         Input data to be transformed
#     transform : str
#         Type of power transform to be applied. Options are 'box-cox' and 'yeo-johnson'

#     Returns
#     -------
#     Union[pd.Series, pl.Series]
#         Transformed data
#     """
#     if transform not in ['box-cox', 'yeo-johnson']:
#         raise ValueError("Invalid transform type. Options are 'box-cox' and 'yeo-johnson'")
#     if (to_pd_s(x).min() <= 0) and (to_pd_s(x).max() >= 0):
#         raise ValueError("Data contains zero or negative values. Power transform is not applicable -- cannot reflect or log-transform negative values")

#     # Convert to pandas
#     x_ = to_pd_s(x)

#     # Check the skewness of the data
#     skew = x_.skew()

#     #

#     pt = PowerTransformer(method=transform)
#     pt.fit(x_.values.reshape(-1, 1))
