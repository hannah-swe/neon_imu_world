from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neon_imu.loaders import load_imu_csv
from neon_imu.transforms import (
    imu_heading_in_world,
    quaternion_norms,
    transform_imu_to_world,
)

# ----------------------------
# SETTINGS (easy to tweak)
# ----------------------------
RAW_ROOT = Path("data/raw")
SUBJECT_GLOB = "sub-*"
IMU_FILENAME = "imu.csv"

# If True: show plots for each subject
SHOW_PLOTS = True

# If you have many subjects and don't want a plot window per subject,
# set SHOW_PLOTS=False and optionally save plots later.
# ----------------------------

# 1) Discover subject folders
subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

# 2) Iterate subjects (linear, no helper function)
for subject_dir in subject_dirs:
    print(f"Processing {subject_dir.name}")

    # 2.1) Locate IMU CSV
    imu_path = subject_dir / IMU_FILENAME
    if not imu_path.exists():
        print(f"  SKIP: IMU file not found: {imu_path}")
        continue

    # 2.2) Load IMU CSV into DataFrame
    df = load_imu_csv(imu_path)

    # 2.3) Extract quaternions in correct order: [w, x, y, z]
    q_wxyz = df[["quaternion w", "quaternion x", "quaternion y", "quaternion z"]].to_numpy(dtype=float)

    # 2.4) Quaternion norm sanity check
    norms = quaternion_norms(q_wxyz)
    norm_min = float(norms.min())
    norm_mean = float(norms.mean())
    norm_max = float(norms.max())
    print(f"  Quaternion norm min/mean/max: {norm_min:.6f} / {norm_mean:.6f} / {norm_max:.6f}")

    bad_idx = np.where(np.abs(norms - 1.0) > 1e-2)[0]  # tolerance 0.01
    print(f"  Samples with |norm-1| > 0.01: {len(bad_idx)}")

    # 2.5) Build time axis in seconds (relative)
    t_ns = df["timestamp [ns]"].to_numpy(dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9

    # 2.6) Heading vector in world coordinates
    heading_world = imu_heading_in_world(q_wxyz)  # shape (N,3)

    # Optional: normalize heading to unit length (usually already near 1)
    heading_norm = np.linalg.norm(heading_world, axis=1, keepdims=True)
    heading_world_unit = heading_world / np.maximum(heading_norm, 1e-12)

    # 2.7) Optional: accelerations (in g) -> world
    acc_g = df[["acceleration x [g]", "acceleration y [g]", "acceleration z [g]"]].to_numpy(dtype=float)
    acc_world_g = transform_imu_to_world(acc_g, q_wxyz)
    acc_world_mag_g = np.linalg.norm(acc_world_g, axis=1)

    # --- At this point you can inspect:
    # q_wxyz[:5], norms[:5], heading_world_unit[:5], acc_world_g[:5], acc_world_mag_g[:10]
    # and compare with df["yaw [deg]"] etc.

    # 2.8) Plotting (per subject)
    if SHOW_PLOTS:
        # Plot heading components
        plt.figure()
        plt.plot(t_s, heading_world_unit[:, 0], label="heading x")
        plt.plot(t_s, heading_world_unit[:, 1], label="heading y")
        plt.plot(t_s, heading_world_unit[:, 2], label="heading z")
        plt.xlabel("time [s]")
        plt.ylabel("unit heading component")
        plt.title(f"{subject_dir.name} – IMU heading in world")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot acceleration magnitude in g
        plt.figure()
        plt.plot(t_s, acc_world_mag_g)
        plt.xlabel("time [s]")
        plt.ylabel("||acc|| [g]")
        plt.title(f"{subject_dir.name} – Acc magnitude (world) [g]")
        plt.tight_layout()
        plt.show()