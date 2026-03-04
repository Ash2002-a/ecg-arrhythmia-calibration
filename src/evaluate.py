from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

DATA_DIR = Path("data/splits")


FEATURE_COLS = [
    "mean","std","min","max","ptp","energy",
    "abs_mean","dx_max","dx_min","dx_abs_mean",
    "prev_rr","next_rr","hr_prev","rr_ratio"
]


def main():

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    X_train = train[FEATURE_COLS]
    y_train = train["class"]

    X_val = val[FEATURE_COLS]
    y_val = val["class"]

    X_test = test[FEATURE_COLS]
    y_test = test["class"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)

    classes = model.classes_

    y_test_bin = label_binarize(y_test, classes=classes)

    print("\nClasses:", classes)

    # -----------------------
    # AUROC
    # -----------------------

    auroc = roc_auc_score(
        y_test_bin,
        probs,
        average="macro",
        multi_class="ovr"
    )

    print("\nMacro AUROC:", auroc)

    # -----------------------
    # PR AUC
    # -----------------------

    pr_auc = average_precision_score(
        y_test_bin,
        probs,
        average="macro"
    )

    print("Macro PR-AUC:", pr_auc)

    # -----------------------
    # Brier score (for V class)
    # -----------------------

    v_index = list(classes).index("V")

    v_true = (y_test == "V").astype(int)
    v_prob = probs[:, v_index]

    brier = brier_score_loss(v_true, v_prob)

    print("\nBrier Score (Ventricular class):", brier)


if __name__ == "__main__":
    main()