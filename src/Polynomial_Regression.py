import numpy as np

class Polynomial_Regression:
    def __init__(self,order=2, lr=0.001, n_iterations=1000):
        self.order = order
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.expanded_X = None

    def fit(self, X, y):
        # initialization
        n_samples, n_features = X.shape
        self.expanded_X = np.ones((n_samples, 1))
        self.weights = np.zeros(self.order + 1)
        self.bias = 0

        # Expansion of X
        for j in range(1, self.order + 1):
            self.expanded_X = np.append(self.expanded_X, np.power(X, j).reshape(-1, 1), axis = 1)

        # Normalizing the expansion
        self.expanded_X[:, 1:] = (self.expanded_X[:, 1:] - np.mean(self.expanded_X[:, 1:], axis=0))\
                                 / np.std(self.expanded_X[:, 1:], axis=0)

        # Learning
        for _ in range(self.n_iterations):
            y_predicted = np.dot(self.expanded_X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(self.expanded_X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(self.expanded_X, self.weights) + self.bias








