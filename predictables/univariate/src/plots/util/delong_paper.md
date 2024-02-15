# Comparing the Areas Under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach

- Elizabeth R. DeLong
  Quintiles, Inc., 1829 East Franklin Street,
  Chapel Hill, North Carolina 27514, U.S.A.

- David M. DeLong
  SAS Institute, Cary, North Carolina 27511, U.S.A.
  and
- Daniel L. Clarke-Pearson
  Division of Oncology, Department of OBGYN, Duke University Medical Center,
  Durham, North Carolina 27710, U.S.A.

## SUMMARY

Methods of evaluating and comparing the performance of diagnostic tests are of increasing importance as new tests are developed and marketed. When a test is based on an observed variable that lies on a continuous or graded scale, an assessment of the overall value of the test can be made through the use of a receiver operating characteristic (ROC) curve. The curve is constructed by varying the cutpoint used to determine which values of the observed variable will be considered abnormal and then plotting the resulting sensitivities against the corresponding false positive rates. When two or more empirical curves are constructed based on tests performed on the same individuals, statistical analysis on differences between curves must take into account the correlated nature of the data. This paper presents a nonparametric approach to the analysis of areas under correlated ROC curves, by using the theory on generalized U-statistics to generate an estimated covariance matrix.

## 1. Introduction

Methods of evaluating and comparing the performance of diagnostic tests or indices are of increasing importance as new tests or indices are developed or measured. When a test is based on an observed variable that lies on a continuous or graded scale, an assessment of the overall value of the test can be made through the use of a receiver operating characteristic (ROC) curve (Hanley and McNeil, 1982; Metz, 1978). The underlying population curve is theoretically given -by varying the cutpoint used to determine the values of the observed variable to be considered abnormal and then plotting the resulting sensitivities against the corresponding false positive rates. If a test could perfectly discriminate, it would have a value above which the entire abnormal population would fall and below which all normal values would fall (or vice versa). The curve would then pass through the point $(0, 1)$ on the unit grid. The closer an ROC curve comes to this ideal point, the better its discriminating ability. A test with no discriminating ability will produce a curve that follows the diagonal of the grid.

For statistical analysis, a recommended index of accuracy associated with an ROC curve is the area under the curve (Swets and Pickett, 1982). The area under the population ROC curve represents the probability that, when the variable is observed for a randomly selected individual from the abnormal population and a randomly selected individual from the normal population, the resulting values will be in the correct order (e.g., abnormal value higher than the normal value). Generally, parametric assumptions are applied on the distributions of the observed variable in the normal and the abnormal populations. Maximum likelihood programs for estimating the area under the curve and relevant parameters under a binormal model assumption have been widely employed (Dorfman and Alf, 1969; .Metz, 1978; Swets and Pickett, 1982) in order to estimate this area, although these distributions cannot be uniquely determined from the ROC curve. The methodology has been extended (Metz, Wang, and Kronman, 1984) to a "bivariate binormal" model for testing differences between correlated sample ROC curves that arise, for example, when different diagnostic tests are performed on the same individuals.

This paper addresses the nonparametric comparison of areas under correlated ROC curves. When calculated by the trapezoidal rule, the area falling under the points comprising an empirical ROC curve has been shown to be equal to the Mann-Whitney U-statistic for comparing distributions of values from the two samples (Bamber, 1975). Although the trapezoidal rule systematically underestimates the true area (Hanley and McNeil, 1982; Swets and Pickett, 1982) when the number of distinct values taken on by a discrete-valued diagnostic variable is small (say, 5 or 6), it nonetheless produces a meaningful statistic that can be used with confidence when the variable takes on a larger number of values. Hanley and McNeil (1983) use some properties of this nonparametric statistic to compare areas under ROC curves arising from two measures applied to the same individuals. Their approach involves calculating for both the normal and the abnormal sample the correlation between the values of the original measures. The average of the two correlations is used along with the average of the areas under the two curves to arrive at an estimated correlation between the two areas. A table that applies when the average area is at least .70 is given. However, for measures that are not continuous or nearly so, their method relies on Gaussian modeling assumptions for estimating the variances of the two areas. In Section 2 we present an alternative methodology using a more completely nonparametric approach which exploits the properties of the Mann-Whitney statistic. Section 3 presents an example of three correlated ROC curves derived from data on ovarian cancer patients undergoing surgery for bowel obstruction. Three different prognostic indices are evaluated and compared.

## 2. Analysis of Areas Under Correlated ROC Curves

Suppose a sample of $N$ individuals undergo a test for predicting an event of interest or determining presence or absence of a medical condition and that the test is based on a continuous-valued diagnostic variable. We will follow the convention that higher values of the test variable are assumed to be associated with the event of interest, e.g., positive disease status. Also suppose it can be determined by means independent of the test that in of these individuals truly undergo the event or have the condition. Let this group be denoted by $C_l$ and let the group of $n (= N - m)$ individuals who do not have the condition be denoted by $C_2$. Let $X_i$, $i = 1, 2, \ldots, m$ and $Y_j$, $j = 1, 2, \ldots, n$ be the values of the variable on which the diagnostic test is based for members of $C_1$ and $C_2$, respectively. These outcome values can be used to construct an empirical ROC curve for assessing the diagnostic performance of the test. For any real number $z$, let

$$
\begin{equation}
\text{sens}(z) = \frac{1}{m} \sum_{i=1}^m I(X_i \geq z)
\end{equation}
$$

where $I(A) = 1$ if $A$ is true and $0$ otherwise. Also let

$$
\begin{equation}
\text{spec}(z) = \frac{1}{n} \sum_{j=1}^n I(Y < z)
\end{equation}
$$

Then $\text{sens}(z)$ is the empirical sensitivity of a test that is derived by dichotomizing the variable into positive or negative results on the basis of the cutpoint $z$ and $\text{spec}(z)$ is the corresponding empirical specificity. Now, as $z$ varies over the possible values of the variable, the empirical ROC curve is a plot of $\text{sens}(z)$ versus $[1 - \text{spec}(z)]$. Clearly, when $z$ is larger than the largest possible value, the curve passes through $(0, 0)$ and it monotonically increases to the point $(1, 1)$ as $z$ decreases to the smallest possible value. To be informative, the entire curve should lie above the $45^{\circ}$ line where $\text{sens}(z) = 1 - \text{spec}(z)$. Selection of an optimal cutpoint depends on a cost function of sensitivity and specificity.

It has been shown that the area under an empirical ROC curve, when calculated by the trapezoidal rule, is equal to the Mann-Whitney two-sample statistic applied to the two samples $\{X_i \}$ and $\{Y_j \}$. Because the Mann-Whitney statistic is a generalized $U$-statistic statistical analysis regarding the performance of diagnostic tests can be performed by exploiting the general theory for $U$-statistics.

The Mann-Whitney statistic estimates the probability, $\theta$, that a randomly selected observation from the population represented by $C_2$ will be less than or equal to a randomly selected observation from the population represented by $C_1$. It can be computed as the average over a kernel, $\psi$, as

$$
\begin{equation}
\hat{\theta} = \frac{1}{mn} \sum_{j=1}^n \sum_{i=1}^m \psi (X_i, Y_j),
\end{equation}
$$

where

$$
\begin{equation}
\psi (X, Y) = \begin{cases}
1 & \text{if } X > Y \\
\frac{1}{2} & \text{if } X = Y \\
0 & \text{if } X < Y
\end{cases}
\end{equation}
$$

In terms of probabilities, $E(0) = 0 = Pr(Y< X) + \frac{1}{2}Pr(X = Y)$. For continuous distributions,

$$
\begin{equation}
Pr(Y= X) = 0
\end{equation}
$$

Asymptotic normality and an expression for the variance of the Mann-Whitney statistic can be derived from theory developed for generalized $U$-statistics by Hoeffding (1948). Define

$$
\begin{align}
\xi_{210} & = E[\psi (X_i, Y_j)\psi(X_i, Y_k)] - \theta^2 \hspace{0.5cm} & j \ne k \\
\xi_{01} & = E[\psi (X_i, Y_j)\psi(X_{\ell}, Y_j)] - \theta^2 \hspace{0.25cm} & i \ne \ell \\
\xi_{11} & = E[\psi (X_i, Y_j)\psi(X_i, Y_j)] - \theta^2
\end{align}
$$

Then

$$
\begin{equation}
Var(\hat{\theta}) = \frac{(n - 1)\xi_{10} + (m - 1)\xi_{01} }{mn} + \frac{\xi_{11}}{mn}
\end{equation}
$$

Bamber (1975) provides a method of estimating the variance in the context of testing the significance of a single ROC curve. Bamber introduces a quantity $B_{xxy}$, which is the probability that two randomly chosen elements of the population $C_1$ will both be greater than or less than a randomly chosen element of $C_2$, minus the complementary probability that the observation from $C_2$ will be between the two from $C_1$. A similar quantity $B_{yyx}$ is also defined and the variance of $A$ is given in terms of $B_{xxy}$ and $B_{yyx}$. $\text{Var}(\hat{\theta})$ is then estimated by empirically estimating $B_{yyx}$ and $B_{xxy}$. Formula (9) can be shown to be equivalent to Bamber's formula (11), which derives from work of Noether (1967) and applies when $X$ and $Y$ are not necessarily continuous.

Hoeffding's theory extends to a vector of $U$-statistics. Let $\theta = (\hat{\theta^1}, \hat{\theta^2}, \ldots, \hat{\theta^k})$ be a vector of statistics, representing the areas under the ROC curves derived from the readings $\{X_i^r\}$, $\{Y_j^r\}$ $(i = 1,..., m$; $j = 1, ..., n$; $1 < r < k)$ of $k$ different diagnostic measures. Then, similar to (6), (7), and (8) above, define

$$
\begin{align}
\xi_{10}^r & = E[\psi (X_i^r, Y_j^r)\psi(X_i^s, Y_j^s)] - \theta^r\theta^s, \hspace{0.25cm} & j \ne k \\
\xi_{01}^r & = E[\psi (X_i^r, Y_j^r)\psi(X_{\ell}^r, Y_j^r)] - \theta^r\theta^s, \hspace{0.25cm} & i \ne \ell \\
\xi_{11}^r & = E[\psi (X_i^r, Y_j^r)\psi(X_i^r, Y_j^r)] - \theta^r\theta^s
\end{align}
$$

The covariance of the $r$th and $s$th statistic can then be written as

$$
\begin{equation}
\text{Cov}(\hat{\theta^r}, \hat{\theta^s}) = \frac{(n - 1)\xi_{10}^{rs} + (m - 1)\xi_{01}^{rs} }{mn} + \frac{\xi_{11}^{rs}}{mn}
\end{equation}
$$

Sen (1960) has provided a method of structural components to provide consistent estimates of the elements of the variance-covariance matrix of a vector of $U$-statistics. This approach turns out to be equivalent to jackknifing, but is conceptually simpler when dealing with $U$-statistics. We will exploit this methodology to compare the areas under two or more ROC curves. For the $r$th statistic, or, the $X$-components and $Y$-components are defined, respectively, as

$$
\begin{align}
V_{10}^r & = \frac{1}{n} \sum_{j=1}^n \psi (X_i^r, Y_j^r) \hspace{0.5cm} & i = 1, 2, \ldots, m \\
V_{01}^r & = \frac{1}{m} \sum_{i=1}^m \psi (X_i^r, Y_j^r) \hspace{0.5cm} & j = 1, 2, \ldots, n
\end{align}
$$

Also define the $k\times k$ matrix $\mathbf{S}_{10}$ such that the $(r, s)$th element is

$$
\begin{equation}
s_{10}^{rs} = \frac{1}{m-1} \sum_{i=1}^m \left[V_{10}^r(X_i) - \hat{\theta^r}\right]\left[V_{10}^s(X_i) - \hat{\theta^s} \right]
\end{equation}
$$

and similarly $\mathbf{S}_{01}$, which has $(r, s)$th element

$$
\begin{equation}
s_{01}^{rs} = \frac{1}{n-1} \sum_{j=1}^n \left[V_{01}^r(Y_j) - \hat{\theta^r}\right]\left[V_{01}^s(Y_j) - \hat{\theta^s} \right]
\end{equation}
$$

The estimated covariance matrix for the vector of parameter estimates, $\hat{\theta} = (\hat{\theta^1}, \hat{\theta^2}, \ldots, \hat{\theta^k})$ is thus

$$
\begin{equation}
\mathbf{S} = \frac{1}{m}\mathbf{S}_{10} + \frac{1}{n}\mathbf{S}_{01}
\end{equation}
$$

Let $g$ be a real-valued function of $\hat{\theta}$ that has bounded second derivatives in a neighborhood of $\theta$. Combining results from Sen (1960) and Arveson (1969, Theorem 16), it follows that if $\lim_{N\rightarrow \infty} m/n$ is bounded and nonzero, then $N^{1/2} [g(\hat{\theta}) - g(\theta)]$ is asymptotically normally distributed with mean $0$ and variance $\sigma_g^2$, where

$$
\begin{equation}
\sigma_g^2 = \lim_{N\rightarrow \infty} \sum_{j=1}^k \sum_{i=1}^k \frac{\partial g}{\partial \theta^j} \frac{\partial g}{\partial \theta^i} \left( \frac{1}{m} \xi_{10}^{i,j} + \frac{1}{n}\xi_{01}^{i,j} \right)
\end{equation}
$$

Further,

$$
\begin{equation}
s_g^2 = \lim_{N\rightarrow \infty} \sum_{j=1}^k \sum_{i=1}^k \frac{\partial g}{\partial \theta^j} \frac{\partial g}{\partial \theta^i} \left( \frac{1}{m} s_{10}^{i,j} + \frac{1}{n}s_{01}^{i,j} \right)
\end{equation}
$$

is a consistent estimate of $\sigma_g^2$.

When $g$ is simply a linear function, the theory reduces considerably, because the partial derivatives are the constants that comprise the linear function. Thus, for any contrast $L\theta '$, where $L$ is a row vector of coefficients,

$$
\begin{equation}
\frac{L\hat{\theta}' - L\theta' }{\left[ L \left( \frac{1}{m} S_{10} + \frac{1}{n} S_{01} \right) L' \right]^{\frac{1}{2}}}
\end{equation}
$$

has a standard normal distribution. A confidence interval for $L\theta'$ naturally follows.

By a modest generalization of these results, we can also apply any set of linear contrasts to a vector of areas under correlated ROC curves and perform a test of significance on
$L\theta'$ . The test then takes the form

$$
\begin{equation}
(\hat{\theta} - \theta) L'  \left[ L \left( \frac{1}{m}S_{10} + \frac{1}{n}S_{01} \right) L' \right]^{-1} L(\hat{\theta} - \theta)'
\end{equation}
$$

which has a chi-square distribution with degrees of freedom equal to the rank of $LSL'$ . A confidence region can also be constructed.

A computer program written in the SAS language is available from the authors for computing components, covariance matrices, and contrasts. However, as indicated in the next section, the components can be computed easily by hand or by a simple computer program. The components can then be input to any program which computes sums of squares and cross-products in order to obtain the covariance matrix $S$.

## 3. Example

When to perform surgical correction of intestinal obstruction in patients known to have ovarian carcinoma is an unresolved problem. The dilemma centers around determining those patients for whom surgery presents a benefit. Castelado et al. (1981), and other authors have proposed that patients who survive longer than 2 months postoperatively can be declared to have "benefited" from the surgery. Using this criterion, Krebs and Goplerud (1983) devised a preoperative scoring system for use as a screening test in determining a patient's risk for failing to benefit from surgery. The scoring algorithm is presented in Table 1. According to this scoring system, patients with low scores should be good candidates for surgery and those with higher scores should be considered at risk for failing to benefit from surgery.

The following example evaluates the discriminating ability of the proposed screening algorithm on 49 consecutive ovarian cancer patients undergoing correction of intestinal obstruction at Duke University Medical Center. Of the 49 patients, 12 survived more than 2 months postoperatively and could be considered surgical successes; the remaining 37 are considered failures. The Krebs-Goplerud score ($K-G$) is compared against two other preoperatively measured indices: total protein ($TP$) and albumin ($ALB$), both of which are positively associated with the patient's nutritional status. Because $ALB$ is one component of $TP$, these two measures are highly correlated, with a Kendall's tau-$b$ value of .65. Increasing levels of $ALB$ and $TP$ are associated with better nutritional status, whereas increasing levels of $K-G$ are associated with poorer prognosis. Thus, to simplify computations, we transformed by subtracting $K-G$ from $12$, the maximum possible value, so that all indices would prognosticate in the same direction.

### Table 1

#### Krebs-Goplerud scoring system for prognostic parameters in ovarian carcinoma complicated by bowel obstruction

| Parameter                        | Level                                                | Assigned Risk Score |
| -------------------------------- | ---------------------------------------------------- | ------------------- |
| Age (yr)                         | <45                                                  | 0                   |
|                                  | 45-65                                                | 1                   |
|                                  | >65                                                  | 2                   |
| Nutritional status (deprivation) | None or minimal                                      | 0                   |
|                                  | Moderate                                             | 1                   |
|                                  | Severe                                               | 2                   |
| Tumor status                     | No palpable intra-abdominal masses                   | 0                   |
|                                  | Palpable intra-abdominal masses                      | 1                   |
|                                  | Liver involvement or distant metastases              | 2                   |
| Ascites                          | None or mild (asymptomatic, abdomen not distended)   | 0                   |
|                                  | Moderate (abdomen distended)                         | 1                   |
|                                  | Severe (symptomatic, requires frequent paracentesis) | 2                   |
| Previous chemotherapy            | None, or no adequate trial                           | 0                   |
|                                  | Failed single-drug therapy                           | 1                   |
|                                  | Failed combination-drug therapy                      | 2                   |
| Previous radiation therapy       | None                                                 | 0                   |
|                                  | Radiation therapy to pelvis                          | 1                   |
|                                  | Radiation therapy to whole abdomen                   | 2                   |

Figure 1 displays the empirical ROC curves for the three indices. From this figure, it appears that $K-G$ offers little improvement over either $ALB$ or $TP$. The estimated areas under the curves for $K-G$, $ALB$, and $TP$ are $.69$, $.72$, and $.65$, respectively. To analyze and compare these areas, the covariance matrix for the vector of areas is needed. The method of structural components easily produces this matrix. For each of the variables of interest, ($K-G$, $ALB$, $TP$), we can denote by $X_r$ $(r = 1, 2, 3)$ the values associated with success and by $y_r$ $(r = 1, 2, 3)$ the values associated with surgical failures. Then, Or = Pr(Y' < Xr) + iPr( yr = X') and we compute the components individually for each of the three varia If the data are first sorted by the variable of interest, it is a simple matter to calculate for each X the number of Y's less than X (NYLx) and the number of Y's equal to X (NYEQx). The component for X is then NYLx + 'NYEQx. Likewise, for each Y we calculate the number of X's greater than Y (NXGy) and the number of X's equal to Y (NXEQy). The component for Y is NXGy + 4NXEQy.

For this example, there are 12 X's and three variables of interest, so the X-components form a 12 x 3 matrix, V10. The 37 Y's yield a component matrix of dimension 37 x 3, V0o. The 3 x 3 matrices S10 and S0l are then computed as

$$
So1 (Vf0V10 - 120'0)
and
Sol = (V1lVol - 370'0).
$$
