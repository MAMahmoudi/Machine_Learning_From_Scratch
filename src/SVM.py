import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_parameter=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.lambda_parameter = lambda_parameter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                if (y_[idx] * self.predict(x_i) >= 1):
                    self.weights -= self.lr * (2 * self.lambda_parameter * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_parameter * self.weights - np.dot(x_i,y_[idx]))
                    self.bias -= self.lr * y_[idx]



    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)