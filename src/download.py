# src/download.py
from __future__ import annotations

from pathlib import Path
from typing import List
import wfdb
from tqdm import tqdm


# Standard MIT-BIH Arrhythmia Database record list (48 records)
MITDB_RECORDS: List[str] = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119",
    "121","122","123","124",
    "200","201","202","203","205","207","208","209",
    "210","212","213","214","215","217","219",
    "220","221","222","223","228","230","231","232","233","234"
]

def download_mitdb(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # wfdb will cache in its own folder by default, but we want explicit files in out_dir.
    # dl_database writes the entire database to a directory, which is simplest and reproducible.
    print("Downloading MIT-BIH Arrhythmia Database (mitdb) from PhysioNet...")
    wfdb.dl_database("mitdb", dl_dir=str(out_dir))
    print(f"Done. Files saved to: {out_dir.resolve()}")

def sanity_check(out_dir: Path, record: str = "100") -> None:
    # IMPORTANT: when reading local files, use the file path without extension
    rec_path = out_dir / record
    ann_path = out_dir / record

    rec = wfdb.rdrecord(str(rec_path))
    ann = wfdb.rdann(str(ann_path), "atr")

    print("\n=== SANITY CHECK ===")
    print(f"Record: {record}")
    print(f"Sampling rate (fs): {rec.fs}")
    print(f"Signal shape: {rec.p_signal.shape}  (samples, channels)")
    print(f"Channel names: {rec.sig_name}")
    print(f"First 10 annotation sample indices: {ann.sample[:10].tolist()}")
    print(f"First 10 annotation symbols: {ann.symbol[:10]}")

def main() -> None:
    out_dir = Path("data/raw/mitdb")
    download_mitdb(out_dir)
    sanity_check(out_dir, record="100")

if __name__ == "__main__":
    main()