from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/beats.parquet")

def main():
    df = pd.read_parquet(DATA_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["class"].value_counts())

    print("\nNumber of ECG records:")
    print(df["record"].nunique())

    print("\nMissing values per column:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nExample rows:")
    print(df.head())

if __name__ == "__main__":
    main()