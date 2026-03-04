from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

DATA_PATH = Path("data/processed/beats.parquet")


def main():

    # ----------------------------
    # Load dataset
    # ----------------------------

    df = pd.read_parquet(DATA_PATH)

    groups = df["record"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print("\nTrain size:", train_df.shape)
    print("Test size:", test_df.shape)

    # ----------------------------
    # Feature columns
    # ----------------------------

    feature_cols = [
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

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df["class"]
    y_test = test_df["class"]

    # =====================================================
    # MODEL 1 — Logistic Regression
    # =====================================================

    print("\nTraining Logistic Regression...")

    logreg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    logreg_pipeline.fit(X_train, y_train)

    preds = logreg_pipeline.predict(X_test)

    print("\nBalanced Accuracy (Logistic Regression):")
    print(balanced_accuracy_score(y_test, preds))

    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test, preds))

    # =====================================================
    # MODEL 2 — Random Forest
    # =====================================================

    print("\n\nTraining Random Forest...")

    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_pipeline.fit(X_train, y_train)

    rf_preds = rf_pipeline.predict(X_test)

    print("\nBalanced Accuracy (Random Forest):")
    print(balanced_accuracy_score(y_test, rf_preds))

    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, rf_preds))

    print("\nConfusion Matrix (Random Forest):")
    print(confusion_matrix(y_test, rf_preds))


if __name__ == "__main__":
    main()