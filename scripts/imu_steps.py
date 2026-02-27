from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from neon_imu.loaders import load_imu_csv
from neon_imu.transforms import quaternion_norms, transform_imu_to_world


def wrap_deg(a: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to [-180, 180]."""
    return (a + 180) % 360 - 180


# ----------------------------
# SETTINGS (easy to tweak)
# ----------------------------
RAW_ROOT = Path("data/raw")
SUBJECT_GLOB = "sub-*"
IMU_FILENAME = "imu.csv"

SHOW_PLOTS = True
# ----------------------------

# 1) Discover subject folders
subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

# 2) Iterate subjects (linear)
for subject_dir in subject_dirs:
    print("\n" + "=" * 80)
    print(f"Processing {subject_dir.name}")
    print("=" * 80)

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
    print(f"  Quaternion norm min/mean/max: {norms.min():.6f} / {norms.mean():.6f} / {norms.max():.6f}")

    bad_idx = np.where(np.abs(norms - 1.0) > 1e-2)[0]  # tolerance 0.01
    print(f"  Samples with |norm-1| > 0.01: {len(bad_idx)}")

    # 2.5) Build time axis in seconds (relative)
    t_ns = df["timestamp [ns]"].to_numpy(dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9

    # 2.6) CSV yaw (wrapped)
    csv_yaw = df["yaw [deg]"].to_numpy(dtype=float)
    csv_yaw_wrapped = wrap_deg(csv_yaw)

    # --------------------------------------------------------------------
    # Heading definition that matches CSV yaw (your BEST MATCH):
    # axis = -Y in IMU coords, rotated into world,
    # yaw = atan2(world_y, world_x) + 90°, wrapped to [-180, 180]
    # --------------------------------------------------------------------
    heading_neutral_in_imu = np.array([0.0, -1.0, 0.0])  # -Y axis in IMU

    heading_world = transform_imu_to_world(heading_neutral_in_imu, q_wxyz)  # (N,3)
    heading_world_unit = heading_world / np.maximum(
        np.linalg.norm(heading_world, axis=1, keepdims=True), 1e-12
    )

    heading_angle_deg = np.degrees(np.arctan2(heading_world_unit[:, 1], heading_world_unit[:, 0]))
    heading_angle_deg = wrap_deg(heading_angle_deg + 90.0)

    # 2.7) Error between derived heading angle and CSV yaw (should be ~0)
    err = wrap_deg(heading_angle_deg - csv_yaw_wrapped)
    rms = float(np.sqrt(np.mean(err**2)))
    print(f"  Yaw vs heading RMS error: {rms:.8f} deg")

    # 2.8) Accelerations (in g) -> world + magnitude
    acc_g = df[["acceleration x [g]", "acceleration y [g]", "acceleration z [g]"]].to_numpy(dtype=float)
    acc_world_g = transform_imu_to_world(acc_g, q_wxyz)
    acc_world_mag_g = np.linalg.norm(acc_world_g, axis=1)

    # ----------------------------
    # PLOTS
    # ----------------------------
    if SHOW_PLOTS:
        # A) CSV yaw vs derived yaw
        plt.figure()
        plt.plot(t_s, csv_yaw_wrapped, label="CSV yaw [deg]")
        plt.plot(t_s, heading_angle_deg, label="Derived yaw from -Y (+90°) [deg]", alpha=0.85)
        plt.xlabel("time [s]")
        plt.ylabel("deg (wrapped to [-180, 180])")
        plt.title(f"{subject_dir.name} – CSV yaw vs derived yaw")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # B) Error plot
        plt.figure()
        plt.plot(t_s, err)
        plt.xlabel("time [s]")
        plt.ylabel("deg")
        plt.title(f"{subject_dir.name} – Error (derived - CSV), RMS={rms:.8f}°")
        plt.tight_layout()
        plt.show()

        # C) Heading vector components in world
        plt.figure()
        plt.plot(t_s, heading_world_unit[:, 0], label="heading x")
        plt.plot(t_s, heading_world_unit[:, 1], label="heading y")
        plt.plot(t_s, heading_world_unit[:, 2], label="heading z")
        plt.xlabel("time [s]")
        plt.ylabel("unit heading component")
        plt.title(f"{subject_dir.name} – Heading vector (world)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # D) Acceleration magnitude in world (g)
        plt.figure()
        plt.plot(t_s, acc_world_mag_g)
        plt.axhline(1.0, linestyle="--", color="grey")
        plt.xlabel("time [s]")
        plt.ylabel("||acc|| [g]")
        plt.title(f"{subject_dir.name} – Acc magnitude (world) [g]")
        plt.tight_layout()
        plt.show()
