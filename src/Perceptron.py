import  numpy as np

class Perceptron:

    def _unit_step_function(self, x):
        return np.where(x>=0, 1, 0)

    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations= n_iterations
        self.activation_function = self._unit_step_function
        self.weights= None
        self.bias = None

    def fit(self, X, y):
        # initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i>0 else 0 for i in y])

        for _ in range (self.n_iterations):
            for idx, x_i in enumerate(X):
                y_predicted = self.predict(x_i)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        y_predicted = self.activation_function(np.dot(X, self.weights) + self.bias)
        return y_predicted