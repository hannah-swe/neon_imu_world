from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from neon_imu.loaders import load_imu_csv
from neon_imu.transforms import quaternion_norms, transform_imu_to_world
from neon_imu.plot_config import setup_plot_style

setup_plot_style()

"""
This script:
    1. reads and checks orientation (quaternions)
    2. derivatives the rotation from IMU to world coordinates
    3. rotates acceleration to world system

IMU-axis convention:
    x = right <-> left
    y = back <-> front
    z = up <-> down
"""

# Configuration:
RAW_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data/processed")
SUBJECT_GLOB = "sub-*"
IMU_FILENAME = "imu.csv"
SHOW_PLOTS = True
SAVE_PLOTS = True


# Baseline timestamps for relative head movements
BASELINE_TS_NS = {
    "sub-997": 1767524149769973658,
    "sub-998": 1771841527840486602,
    "sub-999": 1771841127283972967
}

# Helper to either wrap the angles from -180 to 180 degrees or get the continuous one (possibly over 180)
def wrap_deg(a: np.ndarray) -> np.ndarray:
    # Wrap angles in degrees to [-180, 180]
    return (a + 180) % 360 - 180

def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    # Unwrap a degree angle timeseries to be continuous
    ang_rad = np.deg2rad(angle_deg)
    ang_unwrapped = np.unwrap(ang_rad) # removes +/-pi jumps
    return np.rad2deg(ang_unwrapped)


# 1) Discover subject folders
subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

# 2) Iterate subjects (linear)
for subject_dir in subject_dirs:
    print(f"Processing {subject_dir.name}")

    # 2.1) Locate IMU CSV and load into dataframe
    imu_path = subject_dir / IMU_FILENAME
    if not imu_path.exists():
        print(f"  SKIP: IMU file not found: {imu_path}")
        continue
    df = load_imu_csv(imu_path)

    # 2.2) Extract quaternions in correct order: [w, x, y, z]; sanity check; quaternion norm should be 1 (length = 1)
    q_wxyz = df[["quaternion w", "quaternion x", "quaternion y", "quaternion z"]].to_numpy(dtype=float)
    norms = quaternion_norms(q_wxyz)
    print(f"  Quaternion norm min/mean/max: {norms.min():.6f} / {norms.mean():.6f} / {norms.max():.6f}")
    bad_idx = np.where(np.abs(norms - 1.0) > 1e-2)[0]  # tolerance 0.01
    print(f"  Samples with |norm-1| > 0.01: {len(bad_idx)}")

    # 2.3) Build time axis in seconds (relative, start (0 s) at first timepoint)
    t_ns = df["timestamp [ns]"].to_numpy(dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9

    # 2.4) Pupil labs csv euler for quality check (yaw = rotation over vertical axis; pitch = over sagital axis;
    # roll = over coronal axis;
    # same information as in quaternion but readable for humans in degrees)
    csv_yaw = df["yaw [deg]"].to_numpy(dtype=float)
    csv_yaw_wrapped = wrap_deg(csv_yaw) # wrap it from -180 to 180 degrees
    csv_roll = df["roll [deg]"].to_numpy(dtype=float)
    csv_roll_wrapped = wrap_deg(csv_roll)
    csv_pitch = df["pitch [deg]"].to_numpy(dtype=float)
    csv_pitch_wrapped = wrap_deg(csv_pitch)

    # 2.5) Baseline timestamp for each subject
    baseline_ts = BASELINE_TS_NS[subject_dir.name]
    baseline_idx = int(np.argmin(np.abs(t_ns - np.int64(baseline_ts))))
    print(" Baseline index:", baseline_idx)
    print(" Baseline timestamp (nearest):", int(t_ns[baseline_idx]))

    # 2.6) Relative orientation via relative quaternions
    # q_rel(t) = inv(q0) * q(t)
    rot_abs = R.from_quat(q_wxyz, scalar_first=True)
    rot0 = rot_abs[baseline_idx]
    rot_rel = rot0.inv() * rot_abs
    q_rel_wxyz = rot_rel.as_quat(scalar_first=True)  # (N,4) in wxyz

    # 2.7) Relative heading and yaw (baseline: timestamp per subject from video data by look)
    # Heading definition that matches CSV yaw:
    # axis = -Y in IMU coords, rotated into world,
    # yaw = atan2(world_y, world_x) + 90°, wrapped to [-180, 180]
    heading_neutral_in_imu = np.array([0.0,  1.0, 0.0])  # +Y ist forward
    # Heading direction relative to baseline orientation
    heading_rel_world = transform_imu_to_world(heading_neutral_in_imu, q_rel_wxyz)
    heading_rel_world_unit = heading_rel_world / np.maximum(
        np.linalg.norm(heading_rel_world, axis=1, keepdims=True), 1e-12
    )
    # csv and derived euler in relative heading position (baseline subtraction)
    # derived yaw relative
    yaw_rel_derived = np.degrees(np.arctan2(
        heading_rel_world_unit[:, 1], heading_rel_world_unit[:, 0]
    ))
    yaw_rel_derived = wrap_deg(yaw_rel_derived - 90.0)
    yaw_rel_derived_unwrapped = np.unwrap(yaw_rel_derived)
    # CSV yaw relative (baseline subtraction)
    yaw_rel_csv = wrap_deg(csv_yaw_wrapped - csv_yaw_wrapped[baseline_idx])
    yaw_rel_csv_unwrapped = unwrap_deg(yaw_rel_csv)
    # derived roll relative
    up_neutral_in_imu = np.array([0.0, 0.0, 1.0])
    up_rel_world = transform_imu_to_world(up_neutral_in_imu, q_rel_wxyz)
    up_rel_world_unit = up_rel_world / np.maximum(
        np.linalg.norm(up_rel_world, axis=1, keepdims=True), 1e-12
    )
    roll_rel_derived = np.degrees(np.arctan2(
        up_rel_world_unit[:, 0],
        up_rel_world_unit[:, 2]
    ))
    roll_rel_derived = wrap_deg(roll_rel_derived)
    roll_rel_derived_unwrapped = np.unwrap(roll_rel_derived)
    # csv roll relative (baseline subtraction)
    roll_rel_csv = wrap_deg(csv_roll_wrapped - csv_roll_wrapped[baseline_idx])
    roll_rel_csv_unwrapped = unwrap_deg(roll_rel_csv)
    # derived pitch relative
    pitch_rel_derived = np.degrees(np.arcsin(heading_rel_world_unit[:, 2]))
    pitch_rel_derived = wrap_deg(pitch_rel_derived)
    pitch_rel_derived_unwrapped = np.unwrap(pitch_rel_derived)
    # csv pitch relative (baseline subtraction)
    pitch_rel_csv = wrap_deg(csv_pitch_wrapped - csv_pitch_wrapped[baseline_idx])
    pitch_rel_csv_unwrapped = unwrap_deg(pitch_rel_csv)

    # 2.8) Error between relative derived heading angle and CSV yaw (should be ~0;
    # is my heading interpretation the same as pupil labs yaw?)
    err_rel = wrap_deg(yaw_rel_derived - yaw_rel_csv)
    rms_rel = float(np.sqrt(np.mean(err_rel**2)))
    print(f"  Relative yaw RMS (derived vs CSV): {rms_rel:.8f} deg")
    err_rel_unwrapped = unwrap_deg(yaw_rel_derived_unwrapped - yaw_rel_csv_unwrapped)
    rms_rel_unwrapped = float(np.sqrt(np.mean(err_rel_unwrapped**2)))
    print(f"  Relative yaw RMS unwrapped (derived vs CSV): {rms_rel_unwrapped:.8f} deg")

    # 2.9) Accelerations (in g) -> world + magnitude (~ 1 g is typical with head held still)
    acc_g = df[["acceleration x [g]", "acceleration y [g]", "acceleration z [g]"]].to_numpy(dtype=float)
    # absolute world
    acc_world_g = transform_imu_to_world(acc_g, q_wxyz)
    acc_world_mag_g = np.linalg.norm(acc_world_g, axis=1)
    # relative-to-baseline world (axes rotate with baseline)
    acc_rel_world_g = transform_imu_to_world(acc_g, q_rel_wxyz)
    acc_rel_world_mag_g = np.linalg.norm(acc_rel_world_g, axis=1)


    # PLOTS
    if SHOW_PLOTS:
        # A) Yaw relative to baseline
        plt.figure()
        plt.axhline(0, linestyle="--", color="grey", alpha=0.8)
        sns.lineplot(x=t_s, y=yaw_rel_csv_unwrapped, label="CSV yaw rel [deg]", linewidth=1.75)
        sns.lineplot(x=t_s, y=yaw_rel_derived_unwrapped, label="Derived yaw rel [deg]", linewidth=1.75)
        plt.xlabel("time [s]")
        plt.ylabel("deg (relative to chosen baseline)")
        plt.title(f"{subject_dir.name} – Relative yaw")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/relative_yaw_baseline.png", dpi=400)
        plt.show()

        # B) Quality check: Error plot (error of plot A)
        plt.figure()
        plt.axhline(0, linestyle="--", color="grey", alpha=0.8)
        sns.lineplot(x=t_s, y=err_rel, linewidth=2.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg")
        plt.title(f"{subject_dir.name} – Error (derived - CSV), RMS={rms_rel:.8f}°")
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/yaw_error.png", dpi=400)
        plt.show()

        # A) pitch relative to baseline
        plt.figure()
        plt.axhline(0, linestyle="--", color="grey", alpha=0.8)
        sns.lineplot(x=t_s, y=pitch_rel_csv_unwrapped, label="CSV pitch rel [deg]", linewidth=1.75)
        sns.lineplot(x=t_s, y=pitch_rel_derived_unwrapped, label="Derived pitch rel [deg]", linewidth=1.75)
        plt.xlabel("time [s]")
        plt.ylabel("deg (relative to chosen baseline)")
        plt.title(f"{subject_dir.name} – Relative pitch")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/relative_pitch_baseline.png", dpi=400)
        plt.show()

        # A) roll relative to baseline
        plt.figure()
        plt.axhline(0, linestyle="--", color="grey", alpha=0.8)
        sns.lineplot(x=t_s, y=roll_rel_csv_unwrapped, label="CSV roll rel [deg]", linewidth=1.75)
        sns.lineplot(x=t_s, y=roll_rel_derived_unwrapped, label="Derived roll rel [deg]", linewidth=1.75)
        plt.xlabel("time [s]")
        plt.ylabel("deg (relative to chosen baseline)")
        plt.title(f"{subject_dir.name} – Relative roll")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/relative_roll_baseline.png", dpi=400)
        plt.show()

        # C) Relative heading directions projected onto horizontal plane
        plt.figure()
        xy = heading_rel_world_unit[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
        xy_unit = xy / np.maximum(xy_norm, 1e-12)
        xy_plot = xy_unit.copy()
        step = max(1, len(xy_unit) // 7000) # limit to max 7000 datapoints
        cmap = sns.color_palette("crest_r", as_cmap=True)
        t_norm = (t_s - t_s.min()) / (t_s.max() - t_s.min() + 1e-12)
        for i in range(0, len(xy_plot), step):
            plt.plot([0, xy_plot[i, 0]], [0, xy_plot[i, 1]], color=cmap(t_norm[i]), alpha=0.1)
        theta = np.linspace(0, 2 * np.pi, 400)
        plt.plot(-np.cos(theta), -np.sin(theta), color="grey", alpha=0.8)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("rel world x")
        plt.ylabel("rel world y")
        plt.title(f"{subject_dir.name} – Heading directions")
        # add fixed colorbar
        ax = plt.gca()
        norm = mcolors.Normalize(vmin=t_s.min(), vmax=t_s.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("time [s]")
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/heading_direction_horizontal_plane.png", dpi=400)
        plt.show()

        # D) Acceleration magnitude in world (g), 1 g is typical with head held still
        plt.figure()
        sns.lineplot(x=t_s, y=acc_world_mag_g, linewidth=1.5)
        plt.axhline(1.0, linestyle="--", color="grey", alpha=0.8)
        plt.axvline(t_s[baseline_idx], linestyle="--", color="grey")
        plt.xlabel("time [s]")
        plt.ylabel("acc [g]")
        plt.title(f"{subject_dir.name} – Acc magnitude [g]")
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/acceleration_magnitude_in_g.png", dpi=400)
        plt.show()

        # E) Relative heading vector components (baseline-world)
        plt.figure()
        plt.axhline(0, linestyle="--", color="grey", alpha=0.8)
        plt.axvline(t_s[baseline_idx], linestyle= "--", color="grey")
        sns.lineplot( x=t_s, y=heading_rel_world_unit[:, 0], label="rel heading x", linewidth=2.5)
        sns.lineplot(x=t_s, y=heading_rel_world_unit[:, 1], label="rel heading y", linewidth=2.5)
        sns.lineplot(x=t_s, y=heading_rel_world_unit[:, 2], label="rel heading z", linewidth=2.5)
        plt.xlabel("time [s]")
        plt.ylabel("unit heading component (relative)")
        plt.title(f"{subject_dir.name} – Heading vector (relative to baseline)")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/heading_vectors_relative.png",dpi=400)
        plt.show()

        # E) Relative euler angles yaw, pitch, roll (baseline-world)
        plt.figure()
        plt.axhline(0, linestyle="--", color="grey", alpha=0.8)
        plt.axvline(t_s[baseline_idx], linestyle= "--", color="grey")
        sns.lineplot( x=t_s, y=yaw_rel_csv_unwrapped, label="rel yaw", linewidth=2.5)
        sns.lineplot(x=t_s, y=roll_rel_csv_unwrapped, label="rel roll", linewidth=2.5)
        sns.lineplot(x=t_s, y=pitch_rel_csv_unwrapped, label="rel pitch", linewidth=2.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg (relative)")
        plt.title(f"{subject_dir.name} – Euler angles (relative to baseline)")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(f"{OUTPUT_ROOT}/plots/{subject_dir.name}/euler_angles_relative.png",dpi=400)
        plt.show()