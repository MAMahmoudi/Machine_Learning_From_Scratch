from src.utility import *
import numpy as np


class Regression:
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent
        for _ in range(self.n_iterations):
            y_predicted = self._Approximation(X, self.weights, self.bias)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _Approximation(self, X, w, b):
        raise NotImplementedError()

    def _predict(self, X, w, b):
        raise NotImplementedError()

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)


############################## Linear_Regression
class Linear_Regression(Regression):

    def _Approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b

######################Logistic_Regression
class Logistic_Regression(Regression):

    def _Approximation(self, X, w, b):
        return sigmoid(np.dot(X, w) + b)

    def _predict(self, X, w, b):
        y_predicted = sigmoid(np.dot(X, w) + b)
        predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return predicted_classes