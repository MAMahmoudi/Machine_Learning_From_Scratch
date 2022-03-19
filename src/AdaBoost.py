import numpy as np

class Decision_Stump:

    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:,self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_classifier=5):
        self.n_classifier = n_classifier

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize weights
        weights = np.full(n_samples, (1/n_samples))
        self.classifiers = []
        for _ in range(self.n_classifier):
            classifier = Decision_Stump()
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    missclassified = weights[y != predictions]
                    error  = np.sum(missclassified)
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        min_error = error
                        classifier.polarity = polarity
                        classifier.threshold = threshold
                        classifier.feature_idx = feature_i
            EPS = 1e-10
            classifier.alpha = 0.5 * np.log((1-error)/(error+EPS))
            predictions = classifier.predict(X)
            weights *= np.exp(-classifier.alpha * y * predictions)
            weights /= np.sum(weights)
            self.classifiers.append(classifier)

    def predict(self, X):
        classifier_predictions = [classifier.alpha * classifier.predict(X) for classifier in self.classifiers]
        y_pred = np.sum(classifier_predictions, axis=0)
        return np.sign(y_pred)
