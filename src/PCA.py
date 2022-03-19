import numpy as np

class PCA:

    def __init__(self, n_component):
        self.n_component = n_component
        self.component = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X-self.mean
        # covariance
        covariance = np.cov(X.T)
        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.component = eigenvectors[0:self.n_component]

    def transform(self, X):
        # projrct data
        X = X-self.mean
        return np.dot(X, self.component.T)