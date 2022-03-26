from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression




def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename, index_col=0).dropna().reset_index()
    df = df.drop(columns=['zipcode', 'id', 'date'])
    df = df.drop(df[df.bedrooms == 0].index)
    df = df.drop(df[df.price <= 0].index)
    df = df.drop(df[df.floors == 0].index)
    df = df.drop(df[df.bathrooms == 0].index)
    df = df.drop(df[df.yr_built <= 1700].index)
    response = df.price
    df.drop(['price'], axis=1)
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col_name, col in X.iteritems():
        correlation = str(y.cov(col) / (y.std() * col.std()))
        plt.title(f"Pearson Correlation: {correlation}")
        plt.xlabel(col_name)
        plt.ylabel('price')
        plt.scatter(x=col, y=y)
        plt.show()
        plt.savefig(f"{output_path}/{col_name}")


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    size = 91
    start = 10
    end = 100
    training_size = np.linspace(start, end, size)
    avg_loss, var_loss = np.zeros(size), np.zeros(size)
    for p in training_size:
        losses = np.zeros(10)
        for i in range(10):
            samples = train_X.sample(frac=p/100)
            results = train_y[samples.index]
            lr = LinearRegression()
            lr.fit(samples, results)
            losses[i] = lr.loss(test_X, test_y)
            # samples = train_X.sample(frac=p / 100)
            # results = train_y[samples.index]
            # reg = LinearRegression().fit(samples, results)
            # score = test_X @ reg.coef_
            # m = score.shape[0]
            # losses[i] = np.sum((test_y - score) ** 2) / m
        avg_loss[int(p) - start] = losses.mean()
        var_loss[int(p) - start] = losses.std()
    conf = 2 * np.asarray(var_loss)
    plt.plot(training_size, avg_loss)
    plt.fill_between(training_size, avg_loss-conf, avg_loss+conf, color='y')
    plt.show()





