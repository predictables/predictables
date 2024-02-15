# def a_function():
#     """Am I able to document this function???"""
#     print("Hi")

# from .src import *
# from .rpt import *
# import pandas as pd
# import numpy as np
# import polars as pl
# from dataclasses import dataclass
# from PredicTables.util import get_column_dtype, to_pd_df, to_pd_s
# from typing import Union, List, Dict, Callable

# @dataclass
# class CorrResult:
#     """
#     A dataclass to store the results of a correlation analysis.
#     """

#     x: Union[pd.Series, pl.Series, np.ndarray, None] = None
#     y: Union[pd.Series, pl.Series, np.ndarray, None] = None
#     x_type: str = ""
#     y_type: str = ""
#     x_name: str = ""
#     y_name: str = ""

#     corr: Union[float, int, None] = None

#     has_x: bool = False
#     has_y: bool = False
#     has_corr: bool = False
#     x_has_type: bool = False
#     y_has_type: bool = False
#     x_has_name: bool = False
#     y_has_name: bool = False

#     def __post_init__(self):
#         """
#         Post-initialization method.
#         """
#         self.has_x = self.x is not None
#         self.has_y = self.y is not None
#         self.has_corr = self.corr is not None
#         self.x_has_type = self.x_type != ""
#         self.y_has_type = self.y_type != ""
#         self.x_has_name = self.x_name != ""
#         self.y_has_name = self.y_name != ""

#         if self.has_x and not self.x_has_type:
#             self.x_type = get_column_dtype(self.x)
#         if self.has_y and not self.y_has_type:
#             self.y_type = get_column_dtype(self.y)

#         if self.has_x and not self.x_has_name:
#             self.x_name = self.x.name
#         if self.has_y and not self.y_has_name:
#             self.y_name = self.y.name

#         if (self.has_x and self.has_y) and (not self.has_corr):
#             self.corr = self._calc_corr()

#     def X(self, x: Union[pd.Series, pl.Series, np.ndarray]):
#         """
#         Set the x attribute.

#         Parameters
#         ----------
#         x : Union[pd.Series, pl.Series, np.ndarray]
#             The x attribute.
#         """
#         self.x = x
#         self.has_x = True
#         self.x_type = get_column_dtype(self.x)
#         self.x_name = self.x.name

#     def Y(self, y: Union[pd.Series, pl.Series, np.ndarray]):
#         """
#         Set the y attribute.

#         Parameters
#         ----------
#         y : Union[pd.Series, pl.Series, np.ndarray]
#             The y attribute.
#         """
#         self.y = y
#         self.has_y = True
#         self.y_type = get_column_dtype(self.y)
#         self.y_name = self.y.name

#     def _calc_corr(
#         self,
#         x: Union[pd.Series, pl.Series, np.ndarray, None] = None,
#         y: Union[pd.Series, pl.Series, np.ndarray, None] = None,
#         **kwargs,
#     ):
#         """
#         Calculate the correlation between two variables.

#         Parameters
#         ----------
#         x : Union[pd.Series, pl.Series, np.ndarray, None], optional
#             Optionally sets the x attribute, by default None, indicating no change.
#         y : Union[pd.Series, pl.Series, np.ndarray, None], optional
#             Optionally sets the y attribute, by default None, indicating no change.
#         **kwargs
#             Optional keyword arguments to be passed to the correlation calculation
# method.

#         Returns
#         -------
#         Union[float, int, None]
#             The correlation between two variables.

#         Raises
#         ------
#         ValueError
#             If the correlation cannot be calculated.

#         Examples
#         --------
#         >>> from PredicTables.corr import CorrResult
#         >>> import pandas as pd
#         >>> import numpy as np
#         >>> x = pd.Series(np.random.normal(size=100))
#         >>> y = pd.Series(np.random.normal(size=100))
#         >>> c = CorrResult()
#         >>> c._calc_corr()
#         ValueError: Cannot calculate correlation between two variables.

#         """
#         # Set x or y if provided
#         if x is not None:
#             self.X(x)
#         if y is not None:
#             self.Y(y)

#         missing_vars = []
#         if not self.has_x:
#             missing_vars.append("x")
#         if not self.has_y:
#             missing_vars.append("y")
#         if self.has_x and self.has_y:
#             missing_vars.append("na")

#         # Error handling
#         if not self.has_x or not self.has_y:
#             raise ValueError(
#                 f"Cannot calculate correlation between two variables:\n"
#                 f"x: {self.x}\n"
#                 f"y: {self.y}\n"
#                 f"Pass {missing_vars} to this method to proceed."
#             )

#         # Handle different cases based on the types of x and y
#         if self.x_type == "continuous" and self.y_type == "continuous":
#             return self._calc_pearson()
#         elif self.x_type == "continuous" and self.y_type == "categorical":
#             return self._calc_anova()
#         elif self.x_type == "continuous" and self.y_type == "binary":
#             return self._calc_point_biserial()
#         elif self.x_type == "categorical" and self.y_type == "categorical":
#             return self._calc_cramers_v()
#         elif self.x_type == "categorical" and self.y_type == "binary":
#             return self._calc_cramers_v()
#         elif self.x_type == "binary" and self.y_type == "binary":
#             return self._calc_phi()
#         else:
#             return None

# class Corr:
#     def __init__(self, X, y):
#         """
#         Initialize a Corr object.

#         Parameters
#         ----------
#         X : pandas.DataFrame
#             The feature matrix. Must contain both continuous, categorical, and binary
# features.
#         y : pandas.Series
#             The target vector. Must be binary.
#         """
#         self.X = X
#         self.y = y
#         self.corr = {}
