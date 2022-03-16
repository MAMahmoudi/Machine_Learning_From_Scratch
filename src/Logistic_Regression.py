import numpy as np
from src.utility import *
class Logistic_Regression():
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
        #gradient descent
        for _ in range(self.n_iterations):
            y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)
        predicted_classes = [1 if i>0.5 else 0 for i in y_predicted]
        return predicted_classes

