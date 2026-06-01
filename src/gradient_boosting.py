"""
Gradient Boosting Classifier — from-scratch implementation.

This is the core algorithm that XGBoost and LightGBM are built upon.
We implement the classic Friedman (2001) gradient boosting framework
for binary classification using log-loss (binary cross-entropy).

Algorithm overview
------------------
1. Start with a constant prediction: F₀ = log(p / (1-p))  where p = mean(y)
2. For m = 1 … n_estimators:
   a. Compute pseudo-residuals (negative gradient of log-loss):
      rᵢ = yᵢ - sigmoid(Fᵢ₋₁(xᵢ))
   b. Fit a RegressionTree hₘ to (X, r)
   c. Update: Fₘ(x) = Fₘ₋₁(x) + learning_rate × hₘ(x)
3. Final probability: P(y=1|x) = sigmoid(Fₙ(x))

Key difference from XGBoost
----------------------------
- XGBoost uses *second-order* Taylor expansion (gradients + hessians)
  and adds L1/L2 regularisation on leaf weights.
- LightGBM adds GOSS sampling and histogram-based splitting for speed.
- This implementation uses *first-order* gradients only (classic GBDT).
  It demonstrates the core boosting mechanics without the engineering
  optimisations that require C++ to implement efficiently.

References
----------
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine.
  Annals of Statistics, 29(5), 1189-1232.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
"""

import numpy as np

from src.regression_tree import RegressionTree


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class GradientBoosting:
    """Binary classification via gradient boosted regression trees (GBDT).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees to add sequentially).
    learning_rate : float, default=0.1
        Shrinkage factor applied to each tree's contribution.
        Lower values require more trees but generalise better.
    max_depth : int, default=3
        Maximum depth of each individual regression tree.
        Shallow trees (3-5) work best for boosting.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    subsample : float, default=1.0
        Fraction of training samples used per tree (stochastic GBM).
        Values < 1.0 add randomness and reduce overfitting.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    trees_ : list of RegressionTree
        The trained base learners.
    F0_ : float
        Initial log-odds prediction (constant base model).

    Examples
    --------
    >>> from src.gradient_boosting import GradientBoosting
    >>> import numpy as np
    >>> X = np.random.rand(200, 4)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> gb = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
    >>> gb.fit(X, y)
    >>> gb.predict(X[:5])
    array([...])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: int = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state

        self.trees_: list[RegressionTree] = []
        self.F0_: float = 0.0  # initial log-odds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)  — binary labels {0, 1}
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        y = y.astype(float)
        n_samples = X.shape[0]

        # --- Base model: constant log-odds prediction ---
        p_mean = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
        self.F0_ = np.log(p_mean / (1 - p_mean))

        # Running prediction accumulator (raw log-odds scores)
        F = np.full(n_samples, self.F0_)

        self.trees_ = []

        for _ in range(self.n_estimators):
            # --- Pseudo-residuals (negative gradient of log-loss) ---
            p = _sigmoid(F)
            residuals = y - p  # gradient of log P(y|F) w.r.t. F

            # --- Stochastic subsampling ---
            if self.subsample < 1.0:
                sample_size = max(1, int(n_samples * self.subsample))
                idxs = np.random.choice(n_samples, sample_size, replace=False)
                X_sub, r_sub = X[idxs], residuals[idxs]
            else:
                X_sub, r_sub, idxs = X, residuals, np.arange(n_samples)

            # --- Fit regression tree to pseudo-residuals ---
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sub, r_sub)
            self.trees_.append(tree)

            # --- Update predictions ---
            F += self.learning_rate * tree.predict(X)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates P(y=1|X).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        F = np.full(X.shape[0], self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        prob_positive = _sigmoid(F)
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
