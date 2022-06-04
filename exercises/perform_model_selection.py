from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    func = lambda x: (x+3) * (x+2) * (x+1) * (x-1) * (x-2)
    X = np.linspace(-1.2, 2, n_samples)
    y_no_noise = func(X)
    y = y_no_noise + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.Series(X), pd.Series(y), 2/3)
    train_X, train_y, test_X, test_y = train_X.values, train_y.values, test_X.values, test_y.values
    plt.scatter(train_X, train_y, label='Train Samples')
    plt.scatter(test_X, test_y, label='Test Samples')
    plt.scatter(X, y_no_noise, label='True Model', s=2, c='black')
    plt.title('Train and Test samples drawn from the f(x)+epsilon')
    plt.xlabel('x')
    plt.ylabel('f(x)+epsilon')
    plt.legend()
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.linspace(0, 10, 11).astype(int)
    training_errors = np.zeros(11)
    validation_errors = np.zeros(11)
    for k in degrees:
        p_model = PolynomialFitting(k)
        training_errors[k], validation_errors[k] = cross_validate(p_model, train_X, train_y, mean_square_error)
    plt.plot(degrees, training_errors, label='Training error')
    plt.plot(degrees, validation_errors, label='Validation Error')
    plt.title('Average training and validation error as a function of polynomial degree')
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_errors)
    p_model = PolynomialFitting(k_star).fit(train_X, train_y)
    loss = p_model.loss(test_X, test_y)
    print(str(n_samples) + ' Samples with noise=' + str(noise))
    print("The best degree: ", k_star)
    print("MSE: ", round(loss, 2))
    print("Validation Error: ", np.min(validation_errors))
    print('==========================')




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    proportion = n_samples / len(X)
    train_X, train_y, test_X, test_y = X[:50].values, y[:50].values, X[50:].values, y[50:].values

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas_ridge = np.linspace(0.001, 1, n_evaluations)
    lambdas_lasso = np.linspace(0.001, 1, n_evaluations)
    training_errors_ridge = np.zeros(n_evaluations)
    validation_errors_ridge = np.zeros(n_evaluations)
    training_errors_lasso = np.zeros(n_evaluations)
    validation_errors_lasso = np.zeros(n_evaluations)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i, lam in enumerate(lambdas_ridge):
        ridge = RidgeRegression(lam)
        training_errors_ridge[i], validation_errors_ridge[i] = cross_validate(ridge, train_X, train_y, mean_square_error)
    for i, lam in enumerate(lambdas_lasso):
        lasso = Lasso(alpha=lam, max_iter=10_000)
        training_errors_lasso[i], validation_errors_lasso[i] = cross_validate(lasso, train_X, train_y, mean_square_error)
    ax1.plot(lambdas_ridge, training_errors_ridge, label='Training error')
    ax1.plot(lambdas_ridge, validation_errors_ridge, label='Validation Error')
    ax1.legend()
    ax2.plot(lambdas_lasso, training_errors_lasso, label='Training error')
    ax2.plot(lambdas_lasso, validation_errors_lasso, label='Validation Error')
    ax1.set_title('Ridge')
    ax2.set_title('Lasso')
    ax2.legend()
    fig.supxlabel('Lambda')
    fig.supylabel('Error')
    fig.suptitle('Average training and validation error as a function of Lambda')
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    k_star_ridge = lambdas_ridge[np.argmin(validation_errors_ridge)]
    k_star_lasso = lambdas_lasso[np.argmin(validation_errors_lasso)]

    ridge = RidgeRegression(k_star_ridge).fit(train_X, train_y)
    lasso = Lasso(alpha=k_star_lasso).fit(train_X, train_y)
    linear_regression = LinearRegression().fit(train_X, train_y)

    error_ridge = ridge.loss(test_X, test_y)
    error_lasso = mean_square_error(lasso.predict(test_X), test_y)
    error_linear_regression = linear_regression.loss(test_X, test_y)

    print('Ridge Error: ' + str(round(error_ridge, 2)) + ', Lambda: ' + str(k_star_ridge),
          'Lasso Error: ' + str(round(error_lasso, 2)) + ', Lambda: ' + str(k_star_lasso),
          'Least Squares Error: ' + str(round(error_linear_regression, 2)),
          sep='\n')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
