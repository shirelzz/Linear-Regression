import numpy as np

import Evaluation
import LinearRegressor


def read_data():

    # data = np.loadtxt('prices.txt', delimiter=',')
    #
    # # Split the data into input features (X) and target variable (y)
    # X = data[:, :-1]  # All columns except the last one
    # y = data[:, -1]  # Last column

    data = np.genfromtxt('prices.txt', delimiter=',')

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

    return x_train, x_test, y_train, y_test


def main():

    data = read_data()

    lin_reg = LinearRegressor(0.01)

    x_train, x_test, y_train, y_test = train_test_split(data, 0.25,)

    lin_reg.fit(x_train,y_train)
    predictions = lin_reg.predict(x_test)

    accuracy = Evaluation.accuracy(x_test,predictions)
    recall = Evaluation.recall(x_test,predictions)
    precision = Evaluation.precision(x_test,predictions)

    print("Accuracy: ", accuracy, "\n",
          "Recall: ", recall, "\n",
          "Precision: ", precision)



