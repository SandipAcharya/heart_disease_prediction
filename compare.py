"""
compare.py — Model Comparison: Random Forest vs Gradient Boosting vs XGBoost vs LightGBM

Benchmarks five classifiers on the UCI Heart Disease dataset:

  1. Decision Tree        (from scratch)
  2. Random Forest        (from scratch)
  3. Gradient Boosting    (from scratch — core algorithm behind XGBoost)
  4. XGBoost              (official library — C++ optimised GBDT)
  5. LightGBM             (official library — histogram + leaf-wise GBDT)

Outputs a comparison table + saves a bar chart to assets/model_comparison.png

Usage
-----
    python compare.py
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest
from src.gradient_boosting import GradientBoosting

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH    = "data/heart.csv"
CHART_PATH   = "assets/model_comparison.png"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ---------------------------------------------------------------------------
# Load & split data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["target"]).values
y  = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# Try importing optional libraries
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] xgboost not installed — skipping XGBoost. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARNING] lightgbm not installed — skipping LightGBM. Run: pip install lightgbm")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

models = {
    "Decision Tree\n(scratch)": DecisionTree(max_depth=10, min_samples_split=2),
    "Random Forest\n(scratch)": RandomForest(
        n_trees=50, max_depth=15, min_samples_split=2,
        n_features=None, random_state=RANDOM_STATE
    ),
    "Gradient Boosting\n(scratch)": GradientBoosting(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        subsample=0.8, random_state=RANDOM_STATE
    ),
}

if HAS_XGB:
    models["XGBoost\n(library)"] = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE, verbosity=0
    )

if HAS_LGB:
    models["LightGBM\n(library)"] = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1
    )

# ---------------------------------------------------------------------------
# Benchmark each model
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


results = []

for name, model in models.items():
    label = name.replace("\n", " ")
    _section(label)

    # Train
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Predict
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # AUC (where predict_proba is available)
    auc = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba[:, 1])
        except Exception:
            pass

    results.append({
        "Model": label,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
        "ROC-AUC":   round(auc, 4) if auc else "N/A",
        "Train (s)": round(train_time, 3),
    })

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Train time: {train_time:.3f}s")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["No Disease (0)", "Heart Disease (1)"],
        zero_division=0
    ))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_section("SUMMARY TABLE")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
model_names     = [r["Model"] for r in results]
n_models        = len(results)
n_metrics       = len(metrics_to_plot)

x = np.arange(n_models)
bar_width = 0.18
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 7))
fig.patch.set_facecolor("#1e1e2e")
ax.set_facecolor("#1e1e2e")

for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    values = [r[metric] for r in results]
    bars = ax.bar(x + i * bar_width, values, bar_width,
                  label=metric, color=color, alpha=0.88,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=7.5, color="white", fontweight="bold"
        )

ax.set_xticks(x + bar_width * (n_metrics - 1) / 2)
ax.set_xticklabels(model_names, fontsize=9, color="white")
ax.set_ylim(0.0, 1.12)
ax.set_ylabel("Score", color="white", fontsize=11)
ax.set_title(
    "Model Comparison — Heart Disease Prediction\n"
    "(from-scratch implementations vs library baselines)",
    color="white", fontsize=13, fontweight="bold", pad=14
)
ax.tick_params(colors="white")
ax.spines[:].set_color("#444")
ax.yaxis.grid(True, color="#333", linewidth=0.6, linestyle="--")
ax.set_axisbelow(True)
legend = ax.legend(fontsize=9, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nComparison chart saved -> {CHART_PATH}")

# ---------------------------------------------------------------------------
# Why each model performs as it does — printed analysis
# ---------------------------------------------------------------------------

_section("MODEL ANALYSIS: WHY THE RESULTS DIFFER")
analysis = """
Decision Tree (scratch)
  - Single tree: prone to overfitting or underfitting depending on depth.
  - No ensembling — high variance.  Serves as the baseline.

Random Forest (scratch)
  - Bagging of N trees reduces variance significantly vs a single tree.
  - Random feature sub-sampling decorrelates trees.
  - More stable predictions; typically outperforms single DT.

Gradient Boosting (scratch — core of XGBoost)
  - Sequential boosting: each tree corrects the errors of the previous.
  - First-order gradient (pseudo-residuals) drives learning.
  - Subsample=0.8 adds stochasticity to reduce overfitting.
  - Shallow trees (depth=3) keep bias-variance in balance.
  - This is the exact algorithm XGBoost extends with:
      * Second-order Taylor gradients (hessians) for better leaf weights
      * L1/L2 regularisation on leaf scores
      * Parallel / approximate split finding (C++ speed)


XGBoost (library)
  - Same boosting framework but uses hessian-weighted leaf values:
      w* = -sum(gradients) / (sum(hessians) + lambda)
  - Built-in regularisation prevents overfitting on small datasets.
  - Usually best or tied-best on tabular data.

LightGBM (library)
  - Leaf-wise tree growth (vs level-wise in XGBoost) -> deeper trees faster.
  - GOSS: only high-gradient samples + random low-gradient are used per tree.
  - EFB: sparse features are bundled to reduce feature dimensionality.
  - On small datasets (303 rows) LightGBM's sampling may hurt accuracy;
    it truly shines on millions of rows.

Conclusion for Heart Disease dataset (303 samples)
  - Gradient Boosting / XGBoost typically win on accuracy and AUC.
  - Random Forest is a strong, robust second choice.
  - LightGBM may slightly underperform on tiny datasets but scales best.
"""
print(analysis)
