"""
Regression Tree — variance-reduction CART tree for continuous targets.

Used as the base learner inside GradientBoosting.  Unlike DecisionTree
(which splits on entropy), RegressionTree minimises Mean Squared Error
and assigns each leaf the *mean* of the target values that fall there.
"""

import numpy as np


class RegressionNode:
    """Single node in a regression tree."""

    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left: "RegressionNode" = None,
        right: "RegressionNode" = None,
        *,
        value: float = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # mean of y at this leaf

    def is_leaf_node(self) -> bool:
        return self.value is not None


class RegressionTree:
    """CART regression tree that minimises MSE at each split.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples needed to attempt a split.
    n_features : int or None, default=None
        Number of features to consider per split (None = all).

    Notes
    -----
    Leaf values are the *mean* of the target (pseudo-residuals in
    gradient boosting).  The split criterion is variance reduction:

        gain = var(parent) - (n_L/n)*var(left) - (n_R/n)*var(right)
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        n_features: int = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root: RegressionNode = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_available = X.shape[1]
        self.n_features = (
            n_available if self.n_features is None
            else min(n_available, max(1, self.n_features))
        )
        self.root = self._grow(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> RegressionNode:
        n_samples, n_feats = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return RegressionNode(value=float(np.mean(y)))

        # Random feature sub-sampling
        feat_idxs = np.random.choice(n_feats, min(self.n_features, n_feats), replace=False)
        best_feat, best_thr = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return RegressionNode(value=float(np.mean(y)))

        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return RegressionNode(value=float(np.mean(y)))

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return RegressionNode(best_feat, best_thr, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -np.inf
        best_feat, best_thr = None, None
        parent_var = np.var(y) * len(y)  # total variance (unnormalised)

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue

                gain = parent_var - (np.var(left) * len(left) + np.var(right) * len(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr

    def _traverse(self, x: np.ndarray, node: RegressionNode) -> float:
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
