from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        m = len(X)
        self.mu_ = X.mean()
        self.var_ = np.sum(pow(X - self.mu_, 2))
        if not self.biased_:
            self.var_ /= (m - 1)
        else:
            self.var_ /= m

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return np.array(norm.pdf(X, self.mu_, self.var_))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        val = 1
        m = len(X)
        for i in range(m):
            val *= norm(mu, pow(sigma, 2)).pdf(i)
        return np.log(val)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        m = len(X)
        self.mu_ = X.mean(0)
        centered_X = X - self.mu_
        self.cov_ = np.matmul(np.transpose(centered_X), centered_X) * (1. / m)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return np.array(multivariate_normal.pdf(X, self.mu_, self.cov_))

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        val = 0
        m = len(X)
        d = len(X[0])
        cov_inverted = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        centered_X = X - mu
        centered_X_T = np.transpose(centered_X)
        for i in range(m):
            x = np.matmul(np.matmul(centered_X[i], cov_inverted), centered_X_T[:, i])
            val += np.matmul(np.matmul(centered_X[i], cov_inverted), centered_X_T[:, i])
        val += (m * np.log(cov_det))
        val += (m * d * np.log(2 * np.pi))
        val *= (-0.5)
        return val


if __name__ == '__main__':
    mu, sigma = 10, 1

    # Q1:
    univariate_gaussian = UnivariateGaussian()
    X = np.random.normal(mu, sigma, 1000)
    univariate_gaussian.fit(X)
    print('(', univariate_gaussian.mu_, univariate_gaussian.var_, ')')

    # # Q2:
    ms = np.linspace(10, 1000, 100).astype(int)
    values = []
    for m in ms:
        univariate_gaussian.fit(X[:m])
        values.append(abs(mu - univariate_gaussian.mu_))

    plt.plot(ms, values)
    plt.title('Abs. distance between the estimated and true value of the expectation')
    plt.xlabel('Number of Samples')
    plt.ylabel('Absolute Distance')
    plt.show()

    # Q3:
    Y = univariate_gaussian.pdf(X)
    plt.scatter(X, Y)
    plt.xlabel('samples')
    plt.ylabel('pdf')
    plt.show()

    # Q4:
    mu2 = np.array([0, 0, 4, 0])
    sigma2 = np.array([[1, 0.2, 0, 0.5],
                       [0.2, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0.5, 0, 0, 1]])

    X2 = np.random.multivariate_normal(mu2, sigma2, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(X2)
    print(multivariate_gaussian.mu_, '\n', multivariate_gaussian.cov_)

    # Q5:
    f1, f3 = np.linspace(-10, 10, 30), np.linspace(-10, 10, 30)
    Z = np.array([[multivariate_gaussian.log_likelihood(np.array([i, 0, j, 0]), sigma2, X2) for i in f1] for j in f3])

    graph_ticks = np.linspace(-10,10,20);
    print(graph_ticks)
    ax = sns.heatmap(Z)
    ax.set_xticks = graph_ticks
    ax.set_yticks = graph_ticks
    ax.invert_yaxis()
    plt.show()

    # Q6:
    max_liklihood = np.argmax(Z)
    print('The maximizer of the liklihood is mu=[',
          round(f1[int(max_liklihood / 10)], 3), ', 0 ,',
          round(f3[int(max_liklihood % 10)], 3), ', 0 ]')

