import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"
plt.interactive(False)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=['Date']).dropna().reset_index()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.drop(df[df.Temp <= -50].index)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_X = X.loc[X['Country'] == 'Israel']
    plt.scatter(x=israel_X['DayOfYear'], y=israel_X['Temp'])
    plt.show()
    months = israel_X.groupby('Month').agg({'Temp': 'std'})
    plt.bar(months.index, months['Temp'])
    plt.show()


    # Question 3 - Exploring differences between countries
    fig, ax = plt.subplots()
    country_month = X.groupby(['Country', 'Month']).agg({'Temp': {'mean', 'std'}})
    for country, data in country_month.groupby(level=0):
        data.index.get_level_values('Month')
        plt.errorbar(data.index.get_level_values('Month'), data.Temp['mean'], yerr=data.Temp['std'], label=country)
        plt.legend(fontsize=6)
    plt.show()

    # for country, month in country_month.index:
    #     plt.errorbar(month, country_month.Temp['mean'], yerr=country_month.Temp['std'])
    #     plt.legend(country)
    # plt.show()

    # plt.show()
    # print(country_month)
    # fig, ax = plt.subplots(figsize=(6, 4.5))
    # country_month.plot()
    # plt.show()

    # Question 4 - Fitting model for different values of `k`
    #raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    #raise NotImplementedError()