# import itertools
# from typing import List

# import numpy as np
# import pandas as pd  # type: ignore


# def kernel(x: float, y: float) -> float:
#     """Kernel function. 1 if x > y, 0 if x < y, 0.5 if x == y."""
#     return float(x > y) + 0.5 * float(x == y)


# class ScalarDeLong:
#     """
#     Reproduces the logic and calculation of the DeLong test for comparing two ROC
#     curves. For more information, see:

#     DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more
#     correlated receiver operating characteristic curves: a nonparametric approach.
#     Biometrics. 1988;44:837-845.

#     The order of the step-by-step process in the source should roughly correspond
#     to the order of the methods in this class.

#     """

#     def __init__(self, y: pd.Series, yhat_proba: pd.Series):
#         self.y = y
#         self.yhat_proba = yhat_proba

#     def N(self) -> int:
#         """Returns the number of examples."""
#         return len(self.y)

#     def m(self, z: float) -> int:
#         """Returns the number of positive examples above a given threshold z."""
#         return np.sum(self.y >= z)

#     def n(self, z: float) -> int:
#         """Returns the number of negative examples below a given threshold z."""
#         return np.sum(self.y < z)

#     def X(self, z: float) -> np.ndarray:
#         """Returns the random variable for positive examples, given a threshold z."""
#         return self.yhat_proba[self.y >= z].to_numpy()

#     def Y(self, z: float) -> np.ndarray:
#         """Returns the random variable for negative examples, given a threshold z."""
#         return self.yhat_proba[self.y < z].to_numpy()

#     def sens(self, z: float) -> float:
#         """Returns the sensitivity of the test at a given threshold z."""
#         return float(np.mean(self.X(z) >= z))

#     def spec(self, z: float) -> float:
#         """Returns the specificity of the test at a given threshold z."""
#         return float(np.mean(self.Y(z) < z))

#     def theta_hat(self, z: float) -> float:
#         """Theta represents the probability that a randomly chosen observation from
#         X is greater than a randomly chosen observation from Y."""
#         return sum(kernel(x, y) for x, y in itertools.product(self.X(z), self.Y(z))) / (
#             self.m(z) * self.n(z)
#         )

#     def prob_Y_lt_X(self, z: float) -> float:
#         """Probability that a randomly chosen observation from Y is
#         less than a randomly chosen observation from X.
#         This is the same as 1 - theta_hat."""
#         return float(1 - self.theta_hat(z))

#     def prob_Y_eq_X(self, z: float) -> float:
#         """Probability that a randomly chosen observation from Y is
#         equal to a randomly chosen observation from X."""
#         return float(1 - self.prob_Y_lt_X(z) - self.theta_hat(z))

#     def E_theta_hat(self, z: float) -> float:
#         """Expected value of theta_hat (aka theta itself)."""
#         return self.prob_Y_lt_X(z) + 0.5 * self.prob_Y_eq_X(z)

#     def xi_10(self, z: float) -> float:
#         """Returns E[kernel(X_i, Y_j) * kernel(X_i, Y_k)] - theta^2
#         for i in X and j != k in Y."""
#         ave = float(
#             np.mean(
#                 [
#                     kernel(x, y1) * kernel(x, y2)
#                     for x, y1, y2 in itertools.product(self.X(z), self.Y(z), self.Y(z))
#                     if y1 != y2
#                 ]
#             )
#         )
#         return float(ave - self.theta_hat(z) ** 2)

#     def xi_01(self, z: float) -> float:
#         """Returns E[kernel(X_i, Y_j) * kernel(X_k, Y_j)] - theta^2
#         for i != k in X and j in Y."""
#         ave = float(
#             np.mean(
#                 [
#                     kernel(x1, y) * kernel(x2, y)
#                     for x1, x2, y in itertools.product(self.X(z), self.X(z), self.Y(z))
#                     if x1 != x2
#                 ]
#             )
#         )
#         return float(ave - self.theta_hat(z) ** 2)

#     def xi_11(self, z: float) -> float:
#         """Returns E[kernel(X_i, Y_j) * kernel(X_i, Y_j)] - theta^2
#         for i in X and j in Y."""
#         ave = float(
#             np.mean(
#                 [kernel(x, y) ** 2 for x, y in itertools.product(self.X(z), self.Y(z))]
#             )
#         )
#         return float(ave - self.theta_hat(z) ** 2)

#     def var_theta_hat(self, z: float) -> float:
#         """Returns the variance of theta_hat:

#         (n - 1) xi_10 + (m - 1) xi_01 + xi_11
#         /
#         (m * n)

#         where all values are calculated at a given threshold z, as defined above.
#         """
#         n_term = (self.n(z) - 1) * self.xi_10(z)
#         m_term = (self.m(z) - 1) * self.xi_01(z)

#         numerator = n_term + m_term + self.xi_11(z)
#         denominator = self.m(z) * self.n(z)
#         return numerator / denominator


# class DelongVector:
#     def __init__(self, y: List[pd.Series], yhat_proba: List[pd.Series]):
#         self.y = y
#         self.yhat_proba = yhat_proba

#         if len(y) != len(yhat_proba):
#             raise ValueError(
#                 "The lists y and yhat_proba must be the same length (have the same "
#                 "number of elements)."
#             )

#         self.k = len(y)  # Number of separate ROC curves to compare

#         # Create a list of ScalarDeLong objects for each ROC curve
#         self.d = [ScalarDeLong(y[k], yhat_proba[k]) for k in range(self.k)]

#     def N(self, z: float, k: int) -> int:
#         """Returns the number of examples."""
#         return self.d[k].N()

#     def m(self, z: float, k: int) -> int:
#         """Returns the number of positive examples above a given threshold z."""
#         return self.d[k].m(z)

#     def n(self, z: float, k: int) -> int:
#         """Returns the number of negative examples below a given threshold z."""
#         return self.d[k].n(z)

#     def X(self, z: float, k: int) -> np.ndarray:
#         """Returns the random variable for positive examples, given a threshold z."""
#         return self.d[k].X(z)

#     def Y(self, z: float, k: int) -> np.ndarray:
#         """Returns the random variable for negative examples, given a threshold z."""
#         return self.d[k].Y(z)

#     def sens(self, z: float, k: int) -> float:
#         """Returns the sensitivity of the test at a given threshold z."""
#         return self.d[k].sens(z)

#     def spec(self, z: float, k: int) -> float:
#         """Returns the specificity of the test at a given threshold z."""
#         return self.d[k].spec(z)

#     def theta_hat(self, z: float, k: int) -> float:
#         """Theta represents the probability that a randomly chosen observation from
#         X is greater than a randomly chosen observation from Y."""
#         return self.d[k].theta_hat(z)

#     def prob_Y_lt_X(self, z: float, k: int) -> float:
#         """Probability that a randomly chosen observation from Y is less than a
#         randomly chosen observation from X.
#         This is the same as 1 - theta_hat."""
#         return self.d[k].prob_Y_lt_X(z)

#     def prob_Y_eq_X(self, z: float, k: int) -> float:
#         """Probability that a randomly chosen observation from Y is equal to a randomly
#         chosen observation from X."""
#         return self.d[k].prob_Y_eq_X(z)

#     def E_theta_hat(self, z: float, k: int) -> float:
#         """Expected value of theta_hat (aka theta itself)."""
#         return self.d[k].E_theta_hat(z)

#     def xi_10_rs(self, z: float, r: int, s: int) -> float:
#         """Returns E[kernel(X_i^r, Y_j^r) * kernel(X_i^s, Y_k^s)] -
#         theta^r * theta^s for i in X and j != k in Y."""
#         n_r = len(self.d[r].Y(z))
#         r_count = n_r * (n_r - 1) // 2
#         r_sum = self.d[r].xi_10(z) * r_count

#         n_s = len(self.d[s].Y(z))
#         s_count = n_s * (n_s - 1) // 2
#         s_sum = self.d[s].xi_10(z) * s_count

#         return float((r_sum + s_sum) / (r_count + s_count)) - (
#             self.d[r].theta_hat(z) * self.d[s].theta_hat(z)
#         )

#     def xi_01_rs(self, z: float, r: int, s: int) -> float:
#         """Returns E[kernel(X_i^r, Y_j^r) * kernel(X_k^s, Y_j^s)] -
#         theta^r * theta^s for i != k in X and j in Y."""
#         m_r = len(self.d[r].X(z))
#         r_count = m_r * (m_r - 1) // 2
#         r_sum = self.d[r].xi_01(z) * r_count

#         m_s = len(self.d[s].X(z))
#         s_count = m_s * (m_s - 1) // 2
#         s_sum = self.d[s].xi_01(z) * s_count

#         return float((r_sum + s_sum) / (r_count + s_count)) - (
#             self.d[r].theta_hat(z) * self.d[s].theta_hat(z)
#         )
