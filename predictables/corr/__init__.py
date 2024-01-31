"""
# PredicTables.corr

Defines the PredicTables correlation analysis module. Contains functionality
to calculate correlation (and correlation-like) scores both between features
and between features and the target vector. Also contains functionality to
visualize the results of correlation analysis.

Special functionality is provided for calculating the correlation between (either
two features or a feature and the target vector):

1. Two continuous variables -- Pearson correlation coefficient
2. A continuous feature and a categorical feature -- ANOVA F-value
3. A continuous feature and a binary feature -- Point-biserial correlation coefficient
4. Two categorical features -- Cramer's V
5. A categorical feature and a binary feature -- Cramer's V
6. Two binary features -- Phi coefficient

This module accomplishes this by defining a `Corr` object that contains the attributes
and methods to perform the analysis.

The `Corr` object is instantiated with a feature matrix and a target vector. The
feature matrix can contain both continuous, categorical, and binary features, but
the target vector must be binary. The `Corr` object then calculates the correlation
between each feature and the target vector, and between each pair of features. The
results are stored in the Corr object as a dictionary of dictionaries, where the keys
of the outer dictionary are the names of the features, and the keys of the inner
dictionaries are the names of the features (including the target vector). The values
of the inner dictionaries are the correlation scores.

Example Usage
-------------
>>> from PredicTables.corr import Corr


"""

# from ._corr import Corr  # noqa F401
# from .src import *  # noqa F401
