import numpy as np

class Ridge_Regression:
    def __init__(self, lr=0.001, n_iterations=1000, l2_penality=1):
        self.lr = lr
        self.n_iterations = n_iterations
        self.l2_penality = l2_penality
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (2/n_samples) * (np.dot(X.T, (y_predicted - y)) + (self.l2_penality * self.weights))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
