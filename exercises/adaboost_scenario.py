import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, 250)
    adaboost.fit(train_X, train_y)
    x_axis = np.linspace(1, n_learners, 250).astype(int)
    losses_train = [adaboost.partial_loss(train_X, train_y, t) for t in x_axis]
    losses_test = [adaboost.partial_loss(test_X, test_y, t) for t in x_axis]
    plt.plot(x_axis, losses_train, "b", )
    plt.plot(x_axis, losses_test, "g")
    plt.title('Training and test losses as a function of the number of fitted learners', fontsize=10, y=1.03)
    plt.xlabel('Number of fitted learners')
    plt.ylabel('Loss')
    plt.legend(["Train", "Test"], loc="upper right")
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    for t in T:
        xrange, yrange = np.linspace(*lims[0], 120), np.linspace(*lims[1], 120)
        xx, yy = np.meshgrid(xrange, yrange)
        z = adaboost.partial_predict(np.c_[xx.ravel(), yy.ravel()], t).reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap='Greys')
        plt.colorbar()
        plt.title(f'Decision boundary of T={t}\nTest Error={adaboost.partial_loss(test_X, test_y, t)}')
        plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y,  cmap='spring', alpha=0.8, s=8)
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
        plt.show()

    # Question 3: Decision surface of best performing ensemble
    # Done in q2, this is the last one (T = 250).

    # Question 4: Decision surface with weighted samples
    xrange, yrange = np.linspace(*lims[0], 120), np.linspace(*lims[1], 120)
    xx, yy = np.meshgrid(xrange, yrange)
    z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    size = (adaboost.D_/np.max(adaboost.D_)) * 5
    plt.contourf(xx, yy, z, cmap='Greys')
    plt.colorbar()
    plt.title('Full ensemble decision boundary with size proportional to last weight', fontsize=10, y=1.03)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='spring', alpha=0.8, s=size)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
