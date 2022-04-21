import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data_X file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(f)
        X, y = data[:, [0, 1]], data[:, 2]
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def func(fit: Perceptron, fake_X: np.ndarray, fake_y: int):
            losses.append(fit.loss(X, y))

        perceptron = Perceptron(callback=func)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.plot(losses)
        plt.xlabel('Fitting Iteration')
        plt.ylabel('Loss')
        plt.title('Loss as function of Fitting Iteration')
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    return mu, xs, ys
    # return plt.scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
        # Load dataset
        data = np.load(f)
        X, y = data[:, [0, 1]], data[:, 2]

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_y_pred = lda.predict(X)
        gaussian_naive_bayes = GaussianNaiveBayes()
        gaussian_naive_bayes.fit(X, y)
        gaussian_naive_bayes_y_pred = gaussian_naive_bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        colors = {0.: 'red', 1.: 'green', 2.: 'blue'}
        markers = {0.: '^', 1.: 'o', 2.: 's'}
        figure, (ax1, ax2) = plt.subplots(1, 2)
        gaussian_naive_bayes_loss = round(accuracy(y, gaussian_naive_bayes_y_pred), 3)
        lda_loss = round(accuracy(y, lda_y_pred), 3)
        name = f.split('/')[-1].split('.')[0] + ' dataset'
        ax1.set_title(name + '\n Gaussian Naive Bayes. Accuracy: ' + str(gaussian_naive_bayes_loss), fontsize='small')
        ax2.set_title(name + '\n Lda. Accuracy: ' + str(lda_loss), fontsize='small')

        # Add traces for data-points setting symbols and colors
        for i in range(len(y)):
            ax1.scatter(X[i][0], X[i][1], c=colors[gaussian_naive_bayes_y_pred[i]], marker=markers[y[i]], s=10)
            ax2.scatter(X[i][0], X[i][1], c=colors[lda_y_pred[i]], marker=markers[y[i]], s=10)

        # Add `X` dots specifying fitted Gaussians' means
        ax1.scatter(gaussian_naive_bayes.mu_[:, [0]], gaussian_naive_bayes.mu_[:, [1]], marker='X', c='black', s=20)
        ax2.scatter(lda.mu_[:, [0]], lda.mu_[:, [1]], marker='X', c='black', s=20)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            lda_mu, lda_xs, lda_ys = get_ellipse(lda.mu_[i], lda.cov_)
            gauss_mu, gauss_xs, gauss_ys = get_ellipse(gaussian_naive_bayes.mu_[i], gaussian_naive_bayes.vars_[i])
            ax1.plot(gauss_mu[0] + gauss_xs, gauss_mu[1] + gauss_ys, c="black", linewidth=1)
            ax2.plot(lda_mu[0] + lda_xs, lda_mu[1] + lda_ys, c="black", linewidth=1)
        plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
