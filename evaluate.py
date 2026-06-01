"""
evaluate.py — CLI evaluation script for Heart Disease Prediction.

Trains both the from-scratch Decision Tree and Random Forest on the UCI Heart
Disease dataset, then prints a full classification report for each model.

Usage
-----
    python evaluate.py

Optional environment variable
------------------------------
    HEART_CSV   Path to the CSV file  (default: data/heart.csv)
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.environ.get("HEART_CSV", "data/heart.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["No Disease (0)", "Heart Disease (1)"]
    )

    _section(model_name)
    print(f"  Accuracy : {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Load data ---
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(
        f"Dataset: {len(df)} samples | "
        f"Train: {len(X_train)} | Test: {len(X_test)}"
    )

    # --- Decision Tree ---
    dt = DecisionTree(max_depth=10, min_samples_split=2)
    dt.fit(X_train, y_train)
    _print_report("Decision Tree (from scratch)", y_test, dt.predict(X_test))

    # --- Random Forest ---
    rf = RandomForest(
        n_trees=20,
        max_depth=15,
        min_samples_split=2,
        n_features=None,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    _print_report("Random Forest (from scratch)", y_test, rf.predict(X_test))


if __name__ == "__main__":
    main()
