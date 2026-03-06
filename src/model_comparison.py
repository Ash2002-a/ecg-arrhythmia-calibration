from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

DATA_DIR = Path("data/splits")
FIG_DIR = Path("reports/figures")

FEATURE_COLS = [
    "mean",
    "std",
    "min",
    "max",
    "ptp",
    "energy",
    "abs_mean",
    "dx_max",
    "dx_min",
    "dx_abs_mean",
    "prev_rr",
    "next_rr",
    "hr_prev",
    "rr_ratio"
]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    X_train = train[FEATURE_COLS]
    y_train = train["class"]

    X_test = test[FEATURE_COLS]
    y_test = test["class"]

    # Binary target for ventricular class
    y_test_v = (y_test == "V").astype(int)
    y_train_v = (y_train == "V").astype(int)

    # Logistic Regression
    logreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    # Random Forest
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training Logistic Regression...")
    logreg.fit(X_train, y_train_v)

    print("Training Random Forest...")
    rf.fit(X_train, y_train_v)

    logreg_probs = logreg.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]

    logreg_auc = roc_auc_score(y_test_v, logreg_probs)
    rf_auc = roc_auc_score(y_test_v, rf_probs)

    print(f"Logistic Regression V-class ROC-AUC: {logreg_auc:.4f}")
    print(f"Random Forest V-class ROC-AUC: {rf_auc:.4f}")

    logreg_fpr, logreg_tpr, _ = roc_curve(y_test_v, logreg_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test_v, rf_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(logreg_fpr, logreg_tpr, label=f"Logistic Regression (AUC={logreg_auc:.3f})")
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC={rf_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Ventricular Arrhythmia ROC Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "model_comparison_ventricular_roc.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure: {out_path.resolve()}")


if __name__ == "__main__":
    main()