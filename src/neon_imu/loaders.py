from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_imu_csv(path: str | Path) -> pd.DataFrame:
    """
    Load Pupil Labs Cloud IMU export CSV.

    Returns a DataFrame with the original columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IMU CSV not found: {path.resolve()}")

    df = pd.read_csv(path)

    # Basic expected columns (based on your header)
    required = [
        "timestamp [ns]",
        "quaternion w",
        "quaternion x",
        "quaternion y",
        "quaternion z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in IMU CSV: {missing}")

    return df