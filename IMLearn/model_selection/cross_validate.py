from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    m = X.shape[0]
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)
    training_errors = np.zeros(m)
    validation_errors = np.zeros(m)
    for i in range(len(X_folds)):
        cur_X_folds = np.concatenate(np.delete(X_folds, i, 0))
        cur_y_folds = np.concatenate(np.delete(y_folds, i, 0))
        estimator.fit(cur_X_folds, cur_y_folds)
        training_errors[i] = scoring(estimator.predict(cur_X_folds), cur_y_folds)
        validation_errors[i] = scoring(estimator.predict(X_folds[i]), y_folds[i])
    return training_errors.mean(), validation_errors.mean()

