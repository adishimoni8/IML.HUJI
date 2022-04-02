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
    Load city daily temperature dataset and preprocess data_X.
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
    data_X = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data_X for specific country
    israel_X = data_X.loc[data_X['Country'] == 'Israel']
    plt.figure()
    plt.scatter(x=israel_X.DayOfYear, y=israel_X.Temp, c=israel_X.Year, s=3, cmap="summer")
    plt.colorbar(label="Avg temp' in Israel: 1995-2007", orientation="horizontal")
    plt.show()

    months = israel_X.groupby('Month').agg({'Temp': 'std'})
    plt.bar(months.index, months['Temp'])
    plt.xticks(months.index)
    plt.title('Standard deviation of the daily temperatures with respect to months', y=1.03)
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation')
    plt.show()

    # Question 3 - Exploring differences between countries
    country_month = data_X.groupby(['Country', 'Month']).agg({'Temp': {'mean', 'std'}})
    for country, data in country_month.groupby(level=0):
        data.index.get_level_values('Month')
        plt.errorbar(data.index.get_level_values('Month'), data.Temp['mean'], yerr=data.Temp['std'], label=country)
        plt.legend(fontsize=6)
    plt.title('avg. monthly temp\', with error bars (standard deviation) by contries', y=1.03)
    plt.xlabel('Month')
    plt.ylabel('Avg. Temp\'')
    plt.xticks(country_month.index.get_level_values('Month'))
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    X = israel_X.DayOfYear
    y = israel_X.Temp
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)
    ks = np.linspace(1, 10, 10)
    losses = []
    for k in ks:
        p_model = PolynomialFitting(int(k))
        p_model.fit(train_X, train_y)
        loss = p_model.loss(test_X, test_y)
        losses.append(loss)
    plt.title('Loss of PolyModel for different k\'s', y=1.03)
    plt.xticks(ks)
    plt.xlabel('k')
    plt.ylabel('Loss')
    plt.bar(ks, losses)
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    p_model = PolynomialFitting(k)
    X = israel_X.DayOfYear
    y = israel_X.Temp
    p_model.fit(X, y)

    losses = []
    countries = []
    for country in data_X['Country'].unique():
        if country == 'Israel':
            continue
        countries.append(country)
        country_data = data_X[data_X['Country'] == country]
        country_X, country_y = country_data.DayOfYear, country_data.Temp
        losses.append(p_model.loss(country_X, country_y))
    plt.figure()
    plt.bar(countries, losses)
    plt.title('Loss of israel-trained PolyModel against other contries')
    plt.xlabel('Country')
    plt.ylabel('Loss')
    plt.show()


