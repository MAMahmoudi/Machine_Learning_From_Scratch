import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def is_leaf_node(self):
        return self.value is not None

    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class Decision_Tree:

    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def _grow_tree(self, X, y, depth=0):
        n_sample, n_features = X.shape
        n_labels = len(np.unique(y))
        #stopping criteria
        if (depth >= self.max_depth or n_labels < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        feature_idx = np.random.choice(n_features, self.n_features, replace=False)
        # Greedy search
        best_feature, best_thresh = self._best_criteria(X, y, feature_idx)
        left_idx, right_idx = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idx, :], y[left_idx],depth+1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _traverse_tree(self,x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)


    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_criteria(self, X, y, feature_idx):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feature_idx:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        # Parent entropy
        parent_entropy = entropy(y)
        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # Weighted average child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # return information_gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column <= split_threshold).flatten()
        return left_idxs, right_idxs

    def fit(self, X, y):
        # Grow our tree
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        # Traverse our tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
