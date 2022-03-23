from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, num_of_samples = 10, 1, 1000
    univariate_gaussian = UnivariateGaussian()
    X = np.random.normal(mu, sigma, num_of_samples)
    univariate_gaussian.fit(X)
    print('(' + str(univariate_gaussian.mu_) + ', ' + str(univariate_gaussian.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    values = []
    for m in ms:
        univariate_gaussian.fit(X[:m])
        values.append(abs(mu - univariate_gaussian.mu_))

    plt.plot(ms, values)
    plt.title('Abs. dist. between the estimated and true value of the expectation')
    plt.xlabel('Number of Samples')
    plt.ylabel('Absolute Distance')
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    Y = univariate_gaussian.pdf(X)
    plt.scatter(X, Y, s=4)
    plt.title('The empirical pdf of drawn samples')
    plt.xlabel('samples')
    plt.ylabel('pdf')
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    num_of_samples = 1000
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    X = np.random.multivariate_normal(mu, sigma, num_of_samples)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(X)
    print(multivariate_gaussian.mu_, '\n', multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    mesh_length = 200
    f1, f3 = np.linspace(-10, 10, mesh_length), np.linspace(-10, 10, mesh_length)
    x, y = np.meshgrid(f1, f3)
    z = np.array([[multivariate_gaussian.log_likelihood(np.array([j, 0, i, 0]), sigma, X) for i in f1] for j in f3])
    plt.pcolormesh(x, y, z)
    plt.colorbar()
    plt.show()

    # Question 6 - Maximum likelihood
    maximizer = (np.argwhere(z == z.max()))[0]

    print('The maximizer of the liklihood is mu=[',
          round(f1[maximizer[0]], 4), ', 0 ,',
          round(f3[maximizer[1]], 4), ', 0 ]')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
