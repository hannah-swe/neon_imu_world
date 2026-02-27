from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neon_imu.loaders import load_imu_csv
from neon_imu.transforms import (
    imu_heading_in_world,
    quaternion_norms,
    transform_imu_to_world,
)


def process_subject(subject_dir: Path):
    print(f"\nProcessing {subject_dir.name}")

    imu_path = subject_dir / "imu.csv"
    df = load_imu_csv(imu_path)

    # Extract quaternions in correct order: [w, x, y, z]
    q_wxyz = df[
        ["quaternion w", "quaternion x", "quaternion y", "quaternion z"]
    ].to_numpy(dtype=float)

    # ---- Quaternion norm check
    norms = quaternion_norms(q_wxyz)
    print(f"  Norm min/mean/max: {norms.min():.6f} / {norms.mean():.6f} / {norms.max():.6f}")

    # ---- Heading in world
    heading_world = imu_heading_in_world(q_wxyz)

    # ---- Time axis
    t_ns = df["timestamp [ns]"].to_numpy(dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9

    # ---- Plot heading
    plt.figure()
    plt.plot(t_s, heading_world[:, 0], label="x")
    plt.plot(t_s, heading_world[:, 1], label="y")
    plt.plot(t_s, heading_world[:, 2], label="z")
    plt.title(f"{subject_dir.name} – Heading in World")
    plt.xlabel("time [s]")
    plt.ylabel("heading component")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    raw_root = Path("data/raw")

    subject_dirs = sorted(raw_root.glob("sub-*"))

    if not subject_dirs:
        raise RuntimeError("No sub-* folders found in data/raw")

    for subject_dir in subject_dirs:
        if subject_dir.is_dir():
            process_subject(subject_dir)


if __name__ == "__main__":
    main()
