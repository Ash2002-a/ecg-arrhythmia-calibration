# src/segment.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from .utils import AAMI_MAP, NON_BEAT_SYMBOLS


@dataclass
class SegmentConfig:
    data_dir: Path = Path("data/raw/mitdb")
    out_path: Path = Path("data/processed/beats.parquet")
    record_list: list[str] | None = None

    lead: int = 0              # 0 -> MLII, 1 -> V5 for record 100
    fs: int = 360              # MIT-BIH sampling rate
    pre_s: float = 0.20        # seconds before R
    post_s: float = 0.40       # seconds after R
    drop_q_class: bool = False # set True if you want only N/S/V/F


def list_records(data_dir: Path) -> list[str]:
    # Records are the base names with .hea present
    return sorted([p.stem for p in data_dir.glob("*.hea")])


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x) + 1e-8
    return (x - mu) / sd


def extract_basic_features(window: np.ndarray) -> dict:
    # Simple morphology features (fast, strong baseline)
    dx = np.diff(window, prepend=window[0])
    return {
        "mean": float(np.mean(window)),
        "std": float(np.std(window)),
        "min": float(np.min(window)),
        "max": float(np.max(window)),
        "ptp": float(np.ptp(window)),
        "energy": float(np.sum(window**2)),
        "abs_mean": float(np.mean(np.abs(window))),
        "dx_max": float(np.max(dx)),
        "dx_min": float(np.min(dx)),
        "dx_abs_mean": float(np.mean(np.abs(dx))),
    }


def build_dataset(cfg: SegmentConfig) -> pd.DataFrame:
    data_dir = cfg.data_dir
    records = cfg.record_list or list_records(data_dir)

    pre = int(round(cfg.pre_s * cfg.fs))
    post = int(round(cfg.post_s * cfg.fs))
    win_len = pre + post

    rows = []

    for rec_id in tqdm(records, desc="Segmenting records"):
        rec_path = data_dir / rec_id
        record = wfdb.rdrecord(str(rec_path))
        ann = wfdb.rdann(str(rec_path), "atr")

        sig = record.p_signal[:, cfg.lead].astype(np.float32)
        sig = zscore(sig)  # per-record normalisation (good default)

        r_samples = ann.sample
        symbols = ann.symbol

        # Build RR intervals (needs consecutive beat indices that are real beats)
        beat_indices = []
        beat_symbols = []
        for s_idx, sym in zip(r_samples, symbols):
            if sym in NON_BEAT_SYMBOLS:
                continue
            mapped = AAMI_MAP.get(sym)
            if mapped is None:
                continue
            if cfg.drop_q_class and mapped == "Q":
                continue
            beat_indices.append(int(s_idx))
            beat_symbols.append(mapped)

        beat_indices = np.array(beat_indices, dtype=int)
        beat_symbols = np.array(beat_symbols)

        if len(beat_indices) < 3:
            continue

        # RR features
        prev_rr = np.full(len(beat_indices), np.nan, dtype=np.float32)
        next_rr = np.full(len(beat_indices), np.nan, dtype=np.float32)
        prev_rr[1:] = (beat_indices[1:] - beat_indices[:-1]) / cfg.fs
        next_rr[:-1] = (beat_indices[1:] - beat_indices[:-1]) / cfg.fs

        for i, (r, cls) in enumerate(zip(beat_indices, beat_symbols)):
            start = r - pre
            end = r + post
            if start < 0 or end > len(sig):
                continue

            window = sig[start:end]
            if window.shape[0] != win_len:
                continue

            feats = extract_basic_features(window)

            # RR-derived features (handle edges)
            pr = float(prev_rr[i]) if np.isfinite(prev_rr[i]) else np.nan
            nr = float(next_rr[i]) if np.isfinite(next_rr[i]) else np.nan
            feats.update({
                "prev_rr": pr,
                "next_rr": nr,
                "hr_prev": (60.0 / pr) if pr and np.isfinite(pr) and pr > 0 else np.nan,
                "rr_ratio": (pr / nr) if (np.isfinite(pr) and np.isfinite(nr) and nr > 0) else np.nan,
            })

            rows.append({
                "record": rec_id,
                "r_sample": int(r),
                "class": cls,
                # Store window as bytes to keep parquet compact (optional)
                # If you want raw arrays saved, store as list(window) but file gets big.
                "window": window.astype(np.float32).tobytes(),
                "win_len": win_len,
                **feats,
            })

    df = pd.DataFrame(rows)

    # Drop rows where RR features are NaN (optional)
    # Keep them if you want max data and let model handle NaNs via imputation later.
    return df


def main():
    cfg = SegmentConfig()
    out_dir = cfg.out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(cfg)
    print(df.head())
    print("\nClass counts:\n", df["class"].value_counts())

    df.to_parquet(cfg.out_path, index=False)
    print(f"\nSaved: {cfg.out_path.resolve()}")


if __name__ == "__main__":
    main()