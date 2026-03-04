from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path("data/splits")
FIG_DIR = Path("reports/figures")

FEATURE_COLS = [
    "mean","std","min","max","ptp","energy",
    "abs_mean","dx_max","dx_min","dx_abs_mean",
    "prev_rr","next_rr","hr_prev","rr_ratio"
]

def main():

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_DIR / "train.parquet")

    X_train = train[FEATURE_COLS]
    y_train = train["class"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training model for feature importance...")
    model.fit(X_train, y_train)

    rf = model.named_steps["model"]

    importances = rf.feature_importances_

    df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8,6))

    plt.barh(df["feature"], df["importance"])

    plt.xlabel("Feature Importance")
    plt.title("ECG Feature Importance for Arrhythmia Classification")

    out_path = FIG_DIR / "feature_importance.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)

    print("\nSaved:", out_path.resolve())

if __name__ == "__main__":
    main()