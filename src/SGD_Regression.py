import numpy as np

def learning_schedule(t):
    t0, t1 = 5, 50
    return t0/(t+t1)
class SGD_Regression:
    def __init__(self, lr=0.001, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                random_index = np.random.randint(n_samples)
                X_i = X[random_index:random_index+1]
                y_i = y[random_index:random_index+1]
                y_predicted = np.dot(X_i, self.weights) + self.bias
                dw = 2 * np.dot(X_i.T, (y_predicted - y_i))
                db = (1/n_samples) * np.sum(y_predicted - y_i)
                self.lr = learning_schedule(epoch * n_samples + i)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias








