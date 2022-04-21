from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data_X to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data_X to fit to
        """
        self.classes_ = np.unique(y)
        m, d = X.shape
        self.pi_ = np.zeros(len(self.classes_))
        self.mu_ = np.zeros((len(self.classes_), d))
        self.vars_ = np.zeros((len(self.classes_), d, d))
        for (i, k) in enumerate(self.classes_):
            cur_X = X[np.where(y == k)[0]]
            nk = len(cur_X)
            self.pi_[i] = nk / m
            self.mu_[i] = np.sum(cur_X, axis=0) / nk
            centered_X = cur_X - self.mu_[i]
            self.vars_[i] = (1 / (nk - 1)) * np.diag(np.sum(np.square(centered_X), axis=0))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data_X to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        max_matrix = self.likelihood(X)
        for i in range(len(self.classes_)):
            max_matrix[:, i] *= self.pi_[i]
        return self.classes_[max_matrix.argmax(1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data_X over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data_X to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        m, d = X.shape
        likelihood_matrix = np.zeros((m, len(self.classes_)))
        for i in range(len(self.classes_)):
            det_cov = np.linalg.det(self.vars_[i])
            cov_inv = np.linalg.inv(self.vars_[i])
            Z = np.sqrt((2*np.pi)**d * det_cov)
            centered_X = X - self.mu_[i]
            likelihood_matrix[:, i] = (1/Z) * np.exp(-0.5 * np.einsum('ij,ji->i', centered_X @ cov_inv.T, centered_X.T))
        return likelihood_matrix

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)
