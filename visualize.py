"""
visualize.py — Generate and save a Decision Tree visualization.

Trains a Decision Tree on the UCI Heart Disease dataset and renders the
resulting tree structure as a PNG file (``decision_tree.png``).

Usage
-----
    python visualize.py [--depth DEPTH] [--out PATH]

Arguments
---------
    --depth INT   Maximum depth of the tree to render (default: 4).
    --out   PATH  Output PNG file path (default: decision_tree.png).
"""

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.decision_tree import DecisionTree, Node

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = "data/heart.csv"
DEFAULT_DEPTH = 4
DEFAULT_OUT = "decision_tree.png"

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _get_samples_at_node(
    X: np.ndarray, y: np.ndarray, path: list[tuple]
) -> tuple[np.ndarray, np.ndarray]:
    """Filter (X, y) to only samples that satisfy every condition in *path*.

    Each condition is a ``(feature_idx, threshold, direction)`` tuple where
    ``direction`` is ``"left"`` (``<=``) or ``"right"`` (``>``).
    """
    if not path:
        return X, y
    mask = np.ones(len(X), dtype=bool)
    for feat_idx, threshold, direction in path:
        if direction == "left":
            mask &= X[:, feat_idx] <= threshold
        else:
            mask &= X[:, feat_idx] > threshold
    return X[mask], y[mask]


def visualize_tree(
    root: Node,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_depth: int = DEFAULT_DEPTH,
    out_path: str = DEFAULT_OUT,
) -> None:
    """Render a trained Decision Tree and save it as a PNG.

    Parameters
    ----------
    root : Node
        Root node of the fitted :class:`DecisionTree`.
    X : np.ndarray
        Feature matrix used to compute node statistics.
    y : np.ndarray
        Corresponding target labels.
    feature_names : list of str
        Human-readable feature names.
    max_depth : int
        Maximum depth to render.
    out_path : str
        Destination file path for the output PNG.
    """
    nodes: list[dict] = []
    edges: list[tuple[int, int, str]] = []

    def _traverse(
        node: Node,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        level: int = 0,
        path: list = None,
        parent_id: int = None,
        edge_label: str = "",
    ) -> None:
        if level > max_depth or node is None:
            return
        if path is None:
            path = []

        X_node, y_node = _get_samples_at_node(X, y, path)
        n_samples = len(y_node)
        class_counts = (
            np.bincount(y_node, minlength=2) if n_samples > 0 else np.array([0, 0])
        )

        node_info = {
            "id": len(nodes),
            "x": x_pos,
            "y": y_pos,
            "node": node,
            "n_samples": n_samples,
            "class_counts": class_counts,
        }
        nodes.append(node_info)

        if parent_id is not None:
            edges.append((parent_id, node_info["id"], edge_label))

        if not node.is_leaf_node() and level < max_depth:
            spacing = 4.0 / (2**level)
            left_path = path + [(node.feature, node.threshold, "left")]
            right_path = path + [(node.feature, node.threshold, "right")]
            _traverse(
                node.left, x_pos - spacing, y_pos - 1, level + 1,
                left_path, node_info["id"], "≤"
            )
            _traverse(
                node.right, x_pos + spacing, y_pos - 1, level + 1,
                right_path, node_info["id"], ">"
            )

    _traverse(root)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(18, 12))

    if nodes:
        all_x = [n["x"] for n in nodes]
        all_y = [n["y"] for n in nodes]
        ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
        ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)

        # Draw edges
        for parent_id, child_id, label in edges:
            p, c = nodes[parent_id], nodes[child_id]
            ax.plot(
                [p["x"], c["x"]], [p["y"], c["y"]],
                "k-", alpha=0.5, linewidth=1.5
            )
            mid_x = (p["x"] + c["x"]) / 2
            mid_y = (p["y"] + c["y"]) / 2
            ax.text(
                mid_x, mid_y, label, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        # Draw nodes
        for ni in nodes:
            node = ni["node"]
            n_s = ni["n_samples"]
            cc = ni["class_counts"]
            proportions = cc / n_s if n_s > 0 else np.zeros(2)
            entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])

            if node.is_leaf_node():
                label = (
                    f"Class: {'No Disease' if node.value == 0 else 'Heart Disease'}\n"
                    f"Samples: {n_s}\n"
                    f"Entropy: {entropy:.3f}"
                )
                color = "lightblue" if node.value == 0 else "lightcoral"
            else:
                feat_name = feature_names[node.feature]
                label = (
                    f"{feat_name} ≤ {node.threshold:.3f}\n"
                    f"Samples: {n_s}  |  Entropy: {entropy:.3f}\n"
                    f"[No Disease: {cc[0]}, Heart Disease: {cc[1]}]"
                )
                color = "#f0f0f0"

            ax.text(
                ni["x"], ni["y"], label,
                ha="center", va="center", fontsize=7.5,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=color,
                    edgecolor="#555",
                    linewidth=1.2,
                ),
            )

    ax.axis("off")
    ax.set_title(
        f"Decision Tree Visualization (max_depth={max_depth})",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    legend_elements = [
        mpatches.Patch(color="#f0f0f0", label="Internal Node"),
        mpatches.Patch(color="lightblue", label="Leaf — No Disease"),
        mpatches.Patch(color="lightcoral", label="Leaf — Heart Disease"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Decision tree saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a Decision Tree trained on heart.csv"
    )
    parser.add_argument(
        "--depth", type=int, default=DEFAULT_DEPTH,
        help=f"Max depth to render (default: {DEFAULT_DEPTH})"
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT})"
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    feature_names = df.drop(columns=["target"]).columns.tolist()
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dt = DecisionTree(max_depth=args.depth, min_samples_split=2)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    print(f"Decision Tree accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=["No Disease (0)", "Heart Disease (1)"]
        )
    )

    visualize_tree(
        dt.root, X_train, y_train, feature_names,
        max_depth=args.depth, out_path=args.out
    )


if __name__ == "__main__":
    main()
