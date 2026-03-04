import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

DATA_DIR = Path("data/splits")

FEATURE_COLS = [
    "mean","std","min","max","ptp","energy",
    "abs_mean","dx_max","dx_min","dx_abs_mean",
    "prev_rr","next_rr","hr_prev","rr_ratio"
]

def main():

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    X_train = train[FEATURE_COLS]
    y_train = train["class"]

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

    model.fit(X_train, y_train)

    base_preds = model.predict(X_test)

    base_score = balanced_accuracy_score(y_test, base_preds)

    noise = np.random.normal(0, 0.05, X_test.shape)

    noisy_test = X_test + noise

    noisy_preds = model.predict(noisy_test)

    noisy_score = balanced_accuracy_score(y_test, noisy_preds)

    print("Balanced accuracy (clean):", base_score)
    print("Balanced accuracy (noisy):", noisy_score)

if __name__ == "__main__":
    main()