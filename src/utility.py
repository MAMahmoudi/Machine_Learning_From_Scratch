import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

def sigmoid(X):
     return 1/(1+np.exp(-X))

def accuracy(y_true, y_predicted):
    return np.sum(y_predicted == y_true) / len(y_true)

def plot_learning_curve(model, X_train, X_test, y_train, y_test):
    train_errors, validation_errors = [],[]

    for i in range(1, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        train_errors.append(mse(model.predict(X_train[:i]), y_train[:i]))
        validation_errors.append(mse(model.predict(X_test),y_test))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training")
    plt.plot(np.sqrt(validation_errors), "b-", linewidth=2, label="Validation")
    plt.show()

