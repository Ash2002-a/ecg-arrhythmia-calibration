from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = Path("data/processed/beats.parquet")

def main():
    df = pd.read_parquet(DATA_PATH)
    groups = df["record"]

    # 1) Split out TEST (20% of records)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 2) Split TRAIN/VAL from remaining (VAL = 0.10 of full ≈ 0.10/0.80 = 0.125 of trainval)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df["record"]))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    print("\nShapes:")
    print("Train:", train_df.shape)
    print("Val:  ", val_df.shape)
    print("Test: ", test_df.shape)

    print("\nRecord counts:")
    print("Train records:", train_df["record"].nunique())
    print("Val records:  ", val_df["record"].nunique())
    print("Test records: ", test_df["record"].nunique())

    print("\nClass distribution (Train):")
    print(train_df["class"].value_counts())

    print("\nClass distribution (Val):")
    print(val_df["class"].value_counts())

    print("\nClass distribution (Test):")
    print(test_df["class"].value_counts())

    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    print(f"\nSaved splits to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()