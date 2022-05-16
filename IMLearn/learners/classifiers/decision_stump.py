from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape
        loss = None
        for j in range(d):
            for i in [-1, 1]:
                threshold_, loss_ = self._find_threshold(X[:, j], y, i)
                if loss and loss < loss_:
                    continue
                loss = loss_
                self.threshold_ = threshold_
                self.sign_ = i
                self.j_ = j

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # A nice algorithm I've learned, works for sign = 1, so if the sign is -1 we switch the labels:
        if sign == -1:
            labels = -labels
        # First, we sort X (and y accordingly):
        sort_X_index = np.argsort(values)
        sort_X = values[sort_X_index]
        sort_y = labels[sort_X_index]
        # Then we take the minimizer of the cumulative sum of sort_y_increasing. It is
        # the last place to classify as -1, so we will take one more step:
        min_threshold_ind = np.argmin(np.cumsum(sort_y)) + 1
        # If we're out of boundary:
        if min_threshold_ind >= len(sort_X):
            threshold = sort_X[-1] + 1
        elif min_threshold_ind == 1 and sort_y[0] > 0:
            threshold = sort_X[0]
        else:
            threshold = sort_X[min_threshold_ind]
        # Now, to find the error we need to consider the weights:
        sep_values = np.where(sort_X >= threshold, 1, -1)
        error = np.sum(np.where(np.sign(sort_y) != sep_values, 1, 0) * np.abs(sort_y))
        return threshold, error

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
        y_ = np.where(self.predict(X) > 0, 1, -1)
        return misclassification_error(y, y_)
