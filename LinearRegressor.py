import numpy as np


class LinearRegressor:

    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.slope = 0.0
        self.x = 0.0
        self.b = 0.0
        self.coefficients = None

    def fit(self, x_train, y_train):
        self.x = x_train[:, 0]  # Assuming the first column is the x-values
        self.coefficients = np.zeros(x_train.shape[1] + 1)  # +1 for the bias term

        for _ in range(self.num_iterations):
            self.gradient_descent(x_train, y_train)

    def predict(self, x_test):
        predictions = np.dot(x_test, self.coefficients[1:]) + self.coefficients[0]
        return predictions

    # def loss_function(self, x_train, y_train):
    #     predictions = np.dot(x_train, self.coefficients[1:]) + self.coefficients[0]
    #     error = predictions - y_train
    #     total_error = np.sum(error ** 2)
    #     return total_error / len(x_train)

    def gradient_descent(self, x_train, y_train):
        n = float(len(x_train))
        predictions = np.dot(x_train, self.coefficients[1:]) + self.coefficients[0]
        error = predictions - y_train
        gradient = np.dot(x_train.T, error) / n
        self.coefficients[1:] -= self.learning_rate * gradient
        self.coefficients[0] -= self.learning_rate * np.sum(error) / n

