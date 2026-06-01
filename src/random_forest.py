"""
Random Forest Classifier — from-scratch implementation.

Builds an ensemble of DecisionTree classifiers trained on bootstrap samples
of the training data.  Predictions are made via majority vote.
"""

import numpy as np
from collections import Counter

from src.decision_tree import DecisionTree


class RandomForest:
    """Ensemble of Decision Trees trained with bootstrap aggregation (bagging).

    Each tree is trained on a random bootstrap sample of the training data and
    uses a random subset of features at every split (controlled by ``n_features``).
    Final predictions are made by majority vote across all trees.

    Parameters
    ----------
    n_trees : int, default=10
        Number of decision trees to grow.
    max_depth : int, default=10
        Maximum depth of each individual tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    n_features : int or None, default=None
        Number of features to consider at each split.
        If None, all features are used.
        Common choices: ``int(sqrt(n_features))`` or ``int(log2(n_features))``.
    max_samples : int or None, default=None
        Size of the bootstrap sample drawn for each tree.
        If None, use the full training set size.
    random_state : int or None, default=None
        Seed for NumPy's global random number generator (for reproducibility).

    Attributes
    ----------
    trees : list of DecisionTree
        The trained decision trees after calling :meth:`fit`.

    Examples
    --------
    >>> from src.random_forest import RandomForest
    >>> import numpy as np
    >>> X = np.random.rand(100, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> rf = RandomForest(n_trees=5, max_depth=3, random_state=42)
    >>> rf.fit(X, y)
    >>> preds = rf.predict(X)
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        n_features: int = None,
        max_samples: int = None,
        random_state: int = None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.max_samples = max_samples
        self.trees: list[DecisionTree] = []

        if random_state is not None:
            np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest on the provided data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels (integer-encoded).
        """
        self.trees = []
        n_samples = X.shape[0]

        # Determine bootstrap sample size
        sample_size = (
            n_samples if self.max_samples is None
            else min(self.max_samples, n_samples)
        )

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_samples(X, y, sample_size)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X via majority vote.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # predictions shape: (n_trees, n_samples)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # transpose to (n_samples, n_trees) then majority-vote row-wise
        return np.array(
            [self._most_common_label(row) for row in predictions.T]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bootstrap_samples(X, y, sample_size):
        """Draw a bootstrap sample of size *sample_size* from (X, y)."""
        idxs = np.random.choice(X.shape[0], sample_size, replace=True)
        return X[idxs], y[idxs]

    @staticmethod
    def _most_common_label(y) -> int:
        """Return the majority class label from an array of predictions."""
        return Counter(y).most_common(1)[0][0]
