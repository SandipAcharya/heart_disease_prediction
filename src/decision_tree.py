"""
Decision Tree Classifier — from-scratch implementation using information gain (entropy).

This module provides:
  - Node: internal data structure representing a single tree node.
  - DecisionTree: CART-style binary decision tree for classification.
"""

import numpy as np
from collections import Counter


class Node:
    """Represents a single node in the decision tree.

    A node is either an internal (split) node or a leaf node.
    Leaf nodes store a class label; internal nodes store a split rule.

    Parameters
    ----------
    feature : int or None
        Index of the feature used for splitting (internal nodes only).
    threshold : float or None
        Threshold value for the split (internal nodes only).
    left : Node or None
        Left child node (samples where feature <= threshold).
    right : Node or None
        Right child node (samples where feature > threshold).
    value : int or None
        Predicted class label (leaf nodes only).
    """

    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left: "Node" = None,
        right: "Node" = None,
        *,
        value: int = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """Return True if this node is a leaf (has a class label)."""
        return self.value is not None


class DecisionTree:
    """Binary Decision Tree classifier built from scratch using entropy.

    Grows a CART-style tree by greedily selecting the feature and threshold
    that maximise information gain at each node.

    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    max_depth : int, default=100
        Maximum depth of the tree.  Use a small value to avoid overfitting.
    n_features : int or None, default=None
        Number of features to consider at each split.  If None, all features
        are used.  Useful for random feature sub-sampling (as in RandomForest).

    Attributes
    ----------
    root : Node or None
        Root node of the fitted tree.

    Examples
    --------
    >>> from src.decision_tree import DecisionTree
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTree(max_depth=3)
    >>> tree.fit(X, y)
    >>> tree.predict(X)
    array([0, 0, 1, 1])
    """

    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        n_features: int = None,
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root: Node = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the decision tree to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels (integer-encoded).
        """
        n_available = X.shape[1]
        if self.n_features is None:
            self.n_features = n_available
        else:
            # Clamp to valid range [1, n_available]
            self.n_features = min(n_available, max(1, self.n_features))

        self.root = self._grow_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.root is None:
            return np.zeros(X.shape[0], dtype=int)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively grow the decision tree."""
        n_samples, n_feats = X.shape
        n_unique_labels = len(np.unique(y))

        # --- Stopping criteria ---
        if (
            depth >= self.max_depth
            or n_unique_labels == 1
            or n_samples < self.min_samples_split
        ):
            return Node(value=self._most_common_label(y))

        # --- Random feature sub-sampling ---
        feat_to_select = min(self.n_features, n_feats)
        if feat_to_select <= 0:
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_feats, feat_to_select, replace=False)

        # --- Find best split ---
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # --- Partition data ---
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """Return (feature_index, threshold) giving maximum information gain."""
        best_gain = -1.0
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            for thr in np.unique(X_column):
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold) -> float:
        """Compute information gain for a candidate split."""
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    @staticmethod
    def _split(X_column, split_thresh):
        """Return left and right index arrays based on a threshold."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    @staticmethod
    def _entropy(y) -> float:
        """Shannon entropy of a label array."""
        if len(y) == 0:
            return 0.0
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    @staticmethod
    def _most_common_label(y) -> int:
        """Return the most frequent label in y."""
        if len(y) == 0:
            return 0
        return Counter(y).most_common(1)[0][0]

    def _traverse_tree(self, x, node: Node):
        """Recursively traverse the tree for a single sample."""
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
