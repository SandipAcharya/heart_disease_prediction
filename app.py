"""
Flask web application — Heart Disease Prediction Dashboard.

Serves an interactive UI for tuning Random Forest hyperparameters,
training the model on the UCI Heart Disease dataset, and visualising
performance metrics (confusion matrix, decision boundary).

Usage
-----
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import io
import math
import base64
import traceback

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.decision_tree import DecisionTree  # noqa: F401  (kept for future DT route)
from src.random_forest import RandomForest

# Use non-interactive backend so matplotlib works without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data loading (done once at startup)
# ---------------------------------------------------------------------------

DATA_PATH = "data/heart.csv"

df = pd.read_csv(DATA_PATH)
FEATURE_NAMES: list[str] = df.drop(columns=["target"]).columns.tolist()

X: np.ndarray = df.drop(columns=["target"]).values
y: np.ndarray = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Global slot for the most recently trained model
_trained_model: RandomForest | None = None

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_max_features(value: str, n_features: int) -> int | None:
    """Convert a string ``max_features`` token to an integer (or None).

    Parameters
    ----------
    value : str
        One of ``"None"``, ``"auto"``, ``"sqrt"``, ``"log2"``, or an integer
        string.
    n_features : int
        Total number of features in the dataset.

    Returns
    -------
    int or None
        The resolved feature count, clamped to ``[1, n_features]``.
    """
    if value == "None":
        return None

    if value in ("auto", "sqrt"):
        result = int(math.sqrt(n_features))
    elif value == "log2":
        result = int(math.log2(n_features)) if n_features > 1 else 1
    else:
        try:
            result = int(value)
        except ValueError:
            return None

    return max(1, min(result, n_features))


def _create_bootstrap_sample(
    X: np.ndarray, y: np.ndarray, max_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bootstrap sample of size *max_samples* from (X, y)."""
    n = max(1, min(max_samples, len(X)))
    idxs = np.random.choice(len(X), n, replace=True)
    return X[idxs], y[idxs]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dictionary of classification performance metrics."""
    try:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(
                precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "f1_score": round(
                f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "support": int(len(y_true)),
            "class_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }
    except Exception as exc:
        print(f"[metrics] {exc}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "support": int(len(y_true)),
            "class_report": {},
        }


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Render a confusion-matrix heatmap and return it as a base-64 PNG."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        return _fig_to_base64(fig)
    except Exception as exc:
        print(f"[plot_confusion_matrix] {exc}")
        return ""


def _plot_decision_boundary(
    model: RandomForest,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: tuple[int, int] = (0, 1),
) -> str:
    """Render a 2-D decision-boundary plot and return it as a base-64 PNG.

    All features outside *feature_indices* are fixed at their column mean.
    """
    try:
        fi0, fi1 = feature_indices
        if fi0 == fi1:
            raise ValueError("feature_indices must be two distinct values.")
        if max(fi0, fi1) >= X.shape[1]:
            raise ValueError("feature_indices out of bounds.")

        x_min, x_max = X[:, fi0].min() - 1, X[:, fi0].max() + 1
        y_min, y_max = X[:, fi1].min() - 1, X[:, fi1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1),
        )

        n_grid = xx.ravel().shape[0]
        X_grid = np.tile(np.mean(X, axis=0), (n_grid, 1))
        X_grid[:, fi0] = xx.ravel()
        X_grid[:, fi1] = yy.ravel()

        Z = model.predict(X_grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        sc = ax.scatter(
            X[:, fi0], X[:, fi1], c=y, cmap="coolwarm", edgecolors="k", s=50
        )
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel(FEATURE_NAMES[fi0])
        ax.set_ylabel(FEATURE_NAMES[fi1])
        ax.set_title(
            f"Decision Boundary  ({FEATURE_NAMES[fi0]} vs {FEATURE_NAMES[fi1]})"
        )
        ax.grid(True, alpha=0.3)
        return _fig_to_base64(fig)
    except Exception as exc:
        print(f"[plot_decision_boundary] {exc}")
        return ""


def _fig_to_base64(fig: plt.Figure) -> str:
    """Serialise a matplotlib Figure to a base-64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html", column_names=FEATURE_NAMES)


@app.route("/run_rf", methods=["POST"])
def run_rf():
    """Train a Random Forest with user-supplied hyperparameters.

    Expected form fields
    --------------------
    n_estimators : int   — number of trees (1–200)
    max_depth    : int   — max tree depth (1–50)
    max_features : str   — ``"None"``, ``"auto"``, ``"sqrt"``, ``"log2"``, or int
    bootstrap    : str   — ``"True"`` | ``"False"``
    max_samples  : int   — bootstrap sample size

    Returns
    -------
    JSON with ``success``, ``train_metrics``, ``test_metrics``, ``plots``,
    and ``model_params``.
    """
    global _trained_model

    try:
        # --- Parse parameters ---
        n_estimators = int(request.form.get("n_estimators", 10))
        max_depth = int(request.form.get("max_depth", 15))
        max_features_str = request.form.get("max_features", "None")
        bootstrap = request.form.get("bootstrap", "True") == "True"
        max_samples = int(request.form.get("max_samples", len(X_train)))

        # --- Validate ---
        if not (1 <= n_estimators <= 200):
            return _error("Number of estimators must be between 1 and 200.", 400)
        if not (1 <= max_depth <= 50):
            return _error("Max depth must be between 1 and 50.", 400)
        if max_samples < 1:
            return _error("Max samples must be a positive integer.", 400)

        # --- Resolve max_features ---
        n_features_resolved = _parse_max_features(max_features_str, X_train.shape[1])

        # --- Build training sample ---
        max_samples = max(1, min(max_samples, len(X_train)))
        if bootstrap and max_samples < len(X_train):
            X_tr, y_tr = _create_bootstrap_sample(X_train, y_train, max_samples)
        else:
            X_tr, y_tr = X_train[:max_samples], y_train[:max_samples]

        if len(X_tr) == 0:
            return _error("No valid training samples after filtering.", 400)

        # --- Train ---
        model = RandomForest(
            n_trees=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            n_features=n_features_resolved,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        if len(model.trees) == 0:
            return _error("No trees were successfully trained.", 400)

        _trained_model = model

        # --- Metrics ---
        train_metrics = _compute_metrics(y_tr, model.predict(X_tr))
        test_metrics = _compute_metrics(y_test, model.predict(X_test))

        # --- Plots ---
        confusion_plot = _plot_confusion_matrix(y_test, model.predict(X_test))
        boundary_plot = _plot_decision_boundary(model, X_test, y_test)

        return jsonify(
            {
                "success": True,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "plots": {
                    "decision_boundary": boundary_plot,
                    "confusion_matrix": confusion_plot,
                },
                "model_params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features_str,
                    "bootstrap": bootstrap,
                    "max_samples": max_samples,
                    "actual_training_samples": len(X_tr),
                    "actual_max_features": n_features_resolved,
                    "trees_trained": len(model.trees),
                },
            }
        )

    except Exception as exc:
        print(traceback.format_exc())
        return _error(str(exc), 500)


@app.route("/plot_decision_boundary", methods=["POST"])
def update_decision_boundary():
    """Re-render the decision boundary for a user-selected feature pair.

    Expected form fields
    --------------------
    feature1 : int — index of the first feature
    feature2 : int — index of the second feature
    """
    global _trained_model

    try:
        fi0 = int(request.form.get("feature1"))
        fi1 = int(request.form.get("feature2"))

        if fi0 == fi1:
            return _error("Please select two different features.", 400)
        if not (0 <= fi0 < len(FEATURE_NAMES) and 0 <= fi1 < len(FEATURE_NAMES)):
            return _error("Feature index out of range.", 400)
        if _trained_model is None:
            return _error("No trained model found. Run the algorithm first.", 400)

        plot = _plot_decision_boundary(
            _trained_model, X_test, y_test, feature_indices=(fi0, fi1)
        )
        return jsonify({"success": True, "plot": plot})

    except Exception as exc:
        return _error(str(exc), 500)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _error(message: str, status: int = 400):
    """Return a JSON error response."""
    return jsonify({"success": False, "error": message}), status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
