import numpy as np

import Evaluation
from LinearRegressor import LinearRegressor


def read_data():
    data = np.genfromtxt('prices.txt', delimiter=',')
    # print(data)
    print(data.shape)
    return data


def train_test_split(data, test_size=0.25, random_state=None):
    np.random.seed(random_state) if random_state else None
    np.random.shuffle(data)

    n_samples = data.shape[0]
    n_test = int(n_samples * test_size)

    x_train = data[:-n_test, :-1]
    y_train = data[:-n_test, -1]
    x_test = data[-n_test:, :-1]
    y_test = data[-n_test:, -1]
    # print("x_train: ", x_train, "\n")
    # print("x_test: ", x_test, "\n")
    # print("y_train: ", y_train, "\n")
    # print("y_test: ", y_test, "\n")
    # print("x_train shape: ", x_train.shape, "\n")
    # print("x_test shape: ", x_test.shape, "\n")
    # print("y_train shape: ", y_train.shape, "\n")
    # print("y_test shape: ", y_test.shape, "\n")

    return x_train, x_test, y_train, y_test


def main():

    data = read_data()

    lin_reg = LinearRegressor(0.001, 10000)

    x_train, x_test, y_train, y_test = train_test_split(data, 0.25,)
    print(x_test.shape)

    lin_reg.fit(x_train, y_train)
    predictions = lin_reg.predict(x_test)

    print('y_test: ', y_test)
    print("predictions: ", predictions)

    mse = Evaluation.mean_squared_error(y_test, predictions)
    rs = Evaluation.r_squared(y_test, predictions)
    r2 = Evaluation.root_mean_squared_error(y_test, predictions)

    print("mean_squared_error: ", mse, "\n",
          "r_squared: ", rs, "\n",
          "root_mean_squared_error: ", r2)


if __name__ == '__main__':
    main()


