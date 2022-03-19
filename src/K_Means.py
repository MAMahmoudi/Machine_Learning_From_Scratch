import numpy as np
from src.utility import euclidean_distance
import matplotlib.pyplot as plt

np.random.seed(42)

class K_Means:

    def _create_clusters(self,centroides):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroide(sample, centroides)
            clusters[centroid_idx].append(idx)
        return  clusters

    def _closest_centroide(self, sample, centroides):
        distances = [euclidean_distance(sample, point) for point in centroides]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroid(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return np.sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for clusters_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = clusters_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()


    def __init__(self, k=5, max_iterations=100, plot_steps=False):
        self.k = k
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]
        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        # initialize centroides
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        # optimization
        for _ in range(self.max_iterations):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroid(self.clusters)
            if self.plot_steps:
                self.plot()
            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)