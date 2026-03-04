from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

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
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    X_train = train[FEATURE_COLS]
    y_train = train["class"]

    X_val = val[FEATURE_COLS]
    y_val = val["class"]

    X_test = test[FEATURE_COLS]
    y_test = test["class"]

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("Training base model on TRAIN...")
    base_model.fit(X_train, y_train)

    classes = base_model.classes_
    v_index = list(classes).index("V")

    # ---- Raw probs on test
    raw_probs = base_model.predict_proba(X_test)
    v_true = (y_test == "V").astype(int)
    raw_v_prob = raw_probs[:, v_index]
    raw_brier = brier_score_loss(v_true, raw_v_prob)
    print("\nRaw Brier score (TEST, V class):", raw_brier)

    # ---- Calibrate using VAL only (no retraining of base model)
    # sklearn >= 1.4 uses FrozenEstimator instead of cv="prefit"
    frozen = FrozenEstimator(base_model)

    calibrated = CalibratedClassifierCV(
        estimator=frozen,
        method="sigmoid",   # Platt scaling
        cv=2                # required by API; ignored effectively because estimator is frozen
    )

    print("Fitting calibrator on VAL...")
    calibrated.fit(X_val, y_val)

    cal_probs = calibrated.predict_proba(X_test)
    cal_v_prob = cal_probs[:, v_index]
    cal_brier = brier_score_loss(v_true, cal_v_prob)
    print("Calibrated Brier score (TEST, V class):", cal_brier)

    # ---- Calibration curves
    raw_true, raw_pred = calibration_curve(v_true, raw_v_prob, n_bins=10)
    cal_true, cal_pred = calibration_curve(v_true, cal_v_prob, n_bins=10)

    plt.figure(figsize=(6,6))
    plt.plot(raw_pred, raw_true, marker="o", label="Raw model")
    plt.plot(cal_pred, cal_true, marker="o", label="Platt-calibrated")
    plt.plot([0,1],[0,1],"k--", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Comparison (Ventricular Arrhythmia)")
    plt.legend()

    out_path = FIG_DIR / "calibration_comparison.png"
    plt.savefig(out_path, dpi=300)

    print("\nSaved figure:", out_path.resolve())

if __name__ == "__main__":
    main()