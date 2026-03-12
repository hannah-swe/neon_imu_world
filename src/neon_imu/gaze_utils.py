from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_gaze_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gaze CSV not found: {path.resolve()}")

    df = pd.read_csv(path)

    required = [
        "timestamp [ns]",
        "gaze x [px]",
        "gaze y [px]",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in gaze CSV: {missing}")

    df = df.sort_values("timestamp [ns]").reset_index(drop=True)
    return df


def add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t_ns = df["timestamp [ns]"].to_numpy()
    df["time [s]"] = (t_ns - t_ns[0]) / 1e9
    return df


def subset_gaze_by_time(df: pd.DataFrame, t_start_s: float, t_end_s: float) -> pd.DataFrame:
    if "time [s]" not in df.columns:
        df = add_relative_time(df)

    mask = (df["time [s]"] >= t_start_s) & (df["time [s]"] <= t_end_s)
    return df.loc[mask].copy()


def filter_valid_gaze(df: pd.DataFrame, require_worn: bool = True) -> pd.DataFrame:
    df = df.copy()

    # remove missing gaze points
    df = df.dropna(subset=["gaze x [px]", "gaze y [px]"])

    # optional: only keep samples where tracker was worn
    if require_worn and "worn" in df.columns:
        df = df[df["worn"] == 1]

    # optional: remove blink samples if present
    if "blink id" in df.columns:
        df = df[df["blink id"].isna()]

    return df.reset_index(drop=True)