from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        self.cov_ = np.zeros((d, d))
        for (i, k) in enumerate(self.classes_):
            cur_X = X[np.where(y == k)[0]]
            nk = len(cur_X)
            self.pi_[i] = nk/m
            self.mu_[i] = np.sum(cur_X, axis=0)/nk
            for x in cur_X:
                self.cov_ += np.outer(x - self.mu_[i], x - self.mu_[i].T)
        self.cov_ /= m
        self._cov_inv = np.linalg.inv(self.cov_.T)

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
        det_cov = np.linalg.det(self.cov_)
        likelihood_matrix = np.zeros((m, len(self.classes_)))
        Z = np.sqrt((2 * np.pi) ** d * det_cov)
        for i in range(len(self.classes_)):
            centered_X = X - self.mu_[i]
            likelihood_matrix[:, i] = (1/Z) * np.exp(-0.5 * np.einsum('ij,ji->i', centered_X @ self._cov_inv,
                                                                      centered_X.T))
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
