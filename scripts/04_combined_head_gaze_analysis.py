from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial.transform import Rotation as R

from neon_imu.loaders import load_imu_csv
from neon_imu.plot_config import setup_plot_style
from neon_imu.transforms import (
    interpolate_quaternions_wxyz,
    quaternion_norms,
    transform_imu_to_world,
)
from neon_imu.gaze_transforms import (
    spherical_to_cartesian_scene,
    cartesian_to_spherical_world,
    transform_scene_dirs_to_world,
)

setup_plot_style()


# ----------------------------
# Configuration
# ----------------------------
RAW_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data/processed")
SUBJECT_GLOB = "sub-*"

IMU_FILENAME = "imu.csv"
GAZE_FILENAME = "gaze.csv"

SHOW_PLOTS = True
SAVE_PLOTS = False

BASELINE_TS_NS = {
    "sub-997": 1767524149769973658,
    "sub-998": 1771841527840486602,
    "sub-999": 1771841127283972967,
}

# Time windows per subject (seconds relative to start of gaze.csv)
TIME_WINDOWS_S = {

}
#     "sub-997": (196.0, 210.0),
#     "sub-998": (3.5, 15.5),
#     "sub-999": (1.0, 13.0),

def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180) % 360 - 180


def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

for subject_dir in subject_dirs:
    print(f"Processing {subject_dir.name}")

    imu_path = subject_dir / IMU_FILENAME
    gaze_path = subject_dir / GAZE_FILENAME
    if not imu_path.exists():
        print(f"  SKIP: IMU file not found: {imu_path}")
        continue
    if not gaze_path.exists():
        print(f"  SKIP: Gaze file not found: {gaze_path}")
        continue

    outdir = OUTPUT_ROOT / "plots" / subject_dir.name
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load IMU
    imu_df = load_imu_csv(imu_path).sort_values("timestamp [ns]").reset_index(drop=True)

    q_wxyz = imu_df[["quaternion w", "quaternion x", "quaternion y", "quaternion z"]].to_numpy(dtype=float)
    norms = quaternion_norms(q_wxyz)
    print(f"  Quaternion norm min/mean/max: {norms.min():.6f} / {norms.mean():.6f} / {norms.max():.6f}")

    imu_t_ns = imu_df["timestamp [ns]"].to_numpy(np.int64)
    imu_t_s = (imu_t_ns - imu_t_ns[0]) / 1e9

    # 2) Baseline timestamp -> relative quaternions
    baseline_ts = BASELINE_TS_NS[subject_dir.name]
    baseline_idx = int(np.argmin(np.abs(imu_t_ns - np.int64(baseline_ts))))

    print("  Baseline index:", baseline_idx)
    print("  Baseline timestamp (nearest):", int(imu_t_ns[baseline_idx]))

    rot_abs = R.from_quat(q_wxyz, scalar_first=True)
    rot0 = rot_abs[baseline_idx]
    rot_rel = rot0.inv() * rot_abs
    q_rel_wxyz = rot_rel.as_quat(scalar_first=True)

    # 3) Relative head heading and yaw
    heading_neutral_in_imu = np.array([0.0, 1.0, 0.0])  # +Y = forward
    heading_rel_world = transform_imu_to_world(heading_neutral_in_imu, q_rel_wxyz)
    heading_rel_world_unit = heading_rel_world / np.maximum(
        np.linalg.norm(heading_rel_world, axis=1, keepdims=True), 1e-12
    )
    head_yaw_rel = np.degrees(np.arctan2(
        heading_rel_world_unit[:, 1],
        heading_rel_world_unit[:, 0]
    ))
    head_yaw_rel = wrap_deg(head_yaw_rel - 90.0)
    head_yaw_rel_unwrapped = unwrap_deg(head_yaw_rel)

    head_pitch_rel = np.degrees(np.arctan2(
        heading_rel_world_unit[:, 2],
        np.sqrt(
            heading_rel_world_unit[:, 0] ** 2 +
            heading_rel_world_unit[:, 1] ** 2
        )
    ))

    # 4) Load gaze
    gaze_df = pd.read_csv(gaze_path).sort_values("timestamp [ns]").reset_index(drop=True)

    # optional cleanup
    gaze_df = gaze_df.dropna(subset=["gaze x [px]", "gaze y [px]", "azimuth [deg]", "elevation [deg]"])
    if "worn" in gaze_df.columns:
        gaze_df = gaze_df[gaze_df["worn"] == 1]
    if "blink id" in gaze_df.columns:
        gaze_df = gaze_df[gaze_df["blink id"].isna()]

    gaze_df = gaze_df.reset_index(drop=True)

    gaze_t_ns = gaze_df["timestamp [ns]"].to_numpy(np.int64)
    gaze_t_s = (gaze_t_ns - gaze_t_ns[0]) / 1e9

    gaze_x_px = gaze_df["gaze x [px]"].to_numpy(float)
    gaze_y_px = gaze_df["gaze y [px]"].to_numpy(float)

    gaze_az_scene = gaze_df["azimuth [deg]"].to_numpy(float)
    gaze_el_scene = gaze_df["elevation [deg]"].to_numpy(float)

    t_start_s, t_end_s = TIME_WINDOWS_S.get(subject_dir.name, (gaze_t_s[0], gaze_t_s[-1]))
    time_mask = (gaze_t_s >= t_start_s) & (gaze_t_s <= t_end_s)

    gaze_t_ns = gaze_t_ns[time_mask]
    gaze_t_s = gaze_t_s[time_mask]
    gaze_x_px = gaze_x_px[time_mask]
    gaze_y_px = gaze_y_px[time_mask]
    gaze_az_scene = gaze_az_scene[time_mask]
    gaze_el_scene = gaze_el_scene[time_mask]

    print(f"  Gaze samples in interval [{t_start_s}, {t_end_s}] s: {len(gaze_t_s)}")

    if len(gaze_t_s) == 0:
        print("  SKIP: no gaze samples in interval")
        continue

    # 5) Restrict to the overlap and resample IMU orientation onto gaze timestamps
    overlap_mask = (gaze_t_ns >= imu_t_ns[0]) & (gaze_t_ns <= imu_t_ns[-1])
    num_outside_overlap = int((~overlap_mask).sum())
    if num_outside_overlap:
        print(f"  Dropping {num_outside_overlap} gaze samples outside IMU time span")

    gaze_t_ns = gaze_t_ns[overlap_mask]
    gaze_t_s = gaze_t_s[overlap_mask]
    gaze_x_px = gaze_x_px[overlap_mask]
    gaze_y_px = gaze_y_px[overlap_mask]
    gaze_az_scene = gaze_az_scene[overlap_mask]
    gaze_el_scene = gaze_el_scene[overlap_mask]

    if len(gaze_t_s) == 0:
        print("  SKIP: no overlapping gaze samples after IMU alignment")
        continue

    nearest_idx = np.searchsorted(imu_t_ns, gaze_t_ns, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(imu_t_ns) - 1)
    dt_ms = np.abs(imu_t_ns[nearest_idx] - gaze_t_ns) / 1e6
    print(
        f"  Nearest-sample gaze->IMU dt [ms]: median={np.median(dt_ms):.3f}, max={np.max(dt_ms):.3f}"
    )

    gaze_q_rel_wxyz = interpolate_quaternions_wxyz(imu_t_ns, q_rel_wxyz, gaze_t_ns)

    # 6) Gaze in scene -> relative world
    gaze_scene_dirs = spherical_to_cartesian_scene(gaze_el_scene, gaze_az_scene)

    gaze_rel_world_dirs = transform_scene_dirs_to_world(
        gaze_scene_dirs,
        gaze_q_rel_wxyz,
    )

    gaze_rel_world_dirs = gaze_rel_world_dirs / np.maximum(
        np.linalg.norm(gaze_rel_world_dirs, axis=1, keepdims=True), 1e-12
    )

    # Combined gaze direction in relative world coordinates
    gaze_world_el_rel, gaze_world_az_rel = cartesian_to_spherical_world(gaze_rel_world_dirs)
    gaze_world_az_rel = wrap_deg(gaze_world_az_rel)

    # ----------------------------
    # 6b) Re-baseline combined gaze on the gaze timebase
    # ----------------------------
    baseline_gaze_idx = int(np.argmin(np.abs(gaze_t_ns - np.int64(baseline_ts))))

    print("  Baseline gaze index:", baseline_gaze_idx)
    print("  Baseline gaze timestamp (nearest):", int(gaze_t_ns[baseline_gaze_idx]))

    # Combined gaze relative to gaze-baseline
    gaze_world_az_rel_to_baseline = wrap_deg(
        gaze_world_az_rel - gaze_world_az_rel[baseline_gaze_idx]
    )

    gaze_world_el_rel_to_baseline = (
            gaze_world_el_rel - gaze_world_el_rel[baseline_gaze_idx]
    )

    # 7) Eye contribution (combined gaze - head direction)
    # Interpolate head signals onto the same gaze timestamps used for quaternion resampling
    head_yaw_rel_on_gaze = np.interp(gaze_t_ns.astype(np.float64), imu_t_ns.astype(np.float64), head_yaw_rel)
    head_pitch_rel_on_gaze = np.interp(gaze_t_ns.astype(np.float64), imu_t_ns.astype(np.float64), head_pitch_rel)

    # Approximate eye contribution relative to head
    eye_yaw_rel = wrap_deg(gaze_world_az_rel_to_baseline - head_yaw_rel_on_gaze)
    eye_pitch_rel = gaze_world_el_rel_to_baseline - head_pitch_rel_on_gaze

    # ----------------------------
    # 8) PLOTS
    # ----------------------------
    if SHOW_PLOTS:
        # A) Head yaw vs combined world-gaze yaw
        plt.figure()
        sns.lineplot(x=gaze_t_s, y=head_yaw_rel_on_gaze, label="head yaw rel", linewidth=1.75, alpha=1)
        sns.lineplot(x=gaze_t_s, y=eye_yaw_rel, label="eye yaw rel", linewidth=1.75, alpha=1)
        sns.lineplot(x=gaze_t_s, y=gaze_world_az_rel_to_baseline, label="combined gaze yaw rel", linewidth=1.75, alpha=1)
        plt.axhline(0, linestyle="--", alpha=0.5, color="grey")
        # plt.axvline(imu_t_s[baseline_idx], linestyle="--", alpha=0.5, color="grey")
        plt.xlabel("time [s]")
        plt.ylabel("deg (relative)")
        plt.title(f"{subject_dir.name} – Head and eye yaw vs combined gaze yaw")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(outdir / "combined_head_vs_gaze_yaw.png", dpi=400)
        plt.show()

        # B) Head pitch vs combined world-gaze elevation
        plt.figure()
        sns.lineplot(x=imu_t_s, y=head_pitch_rel, label="head pitch rel [deg]", linewidth=2.5)
        sns.lineplot(x=gaze_t_s, y=gaze_world_el_rel, label="combined gaze elevation rel [deg]", linewidth=2.5)
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.axvline(imu_t_s[baseline_idx], linestyle="--", alpha=0.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg")
        plt.title(f"{subject_dir.name} – Head pitch vs combined gaze elevation")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(outdir / "combined_head_vs_gaze_pitch.png", dpi=400)
        plt.show()

        # C) Eye-only contribution (difference between combined gaze and head)
        plt.figure()
        sns.lineplot(x=gaze_t_s, y=eye_yaw_rel, label="eye yaw rel [deg]", linewidth=2.5)
        sns.lineplot(x=gaze_t_s, y=eye_pitch_rel, label="eye pitch rel [deg]", linewidth=2.5)
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg")
        plt.title(f"{subject_dir.name} – Estimated eye contribution")
        plt.legend()
        sns.despine()
        if SAVE_PLOTS:
            plt.savefig(outdir / "estimated_eye_contribution.png", dpi=400)
        plt.show()

        # D) Combined gaze direction fan in relative world (horizontal plane)
        fig, ax = plt.subplots()
        xy = gaze_rel_world_dirs[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
        xy_unit = xy / np.maximum(xy_norm, 1e-12)
        # only for intuitive plotting: flip both axes if desired
        xy_plot = xy_unit.copy()
        xy_plot[:, 0] *= -1
        xy_plot[:, 1] *= -1
        cmap = sns.color_palette("crest", as_cmap=True)
        t_norm = (gaze_t_s - gaze_t_s.min()) / (gaze_t_s.max() - gaze_t_s.min() + 1e-12)
        step = max(1, len(xy_plot) // 1500)
        for i in range(0, len(xy_plot), step):
            ax.plot([0, xy_plot[i, 0]], [0, xy_plot[i, 1]], color=cmap(t_norm[i]), alpha=0.15)
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(-np.cos(theta), -np.sin(theta), color="grey", alpha=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("relative world x")
        ax.set_ylabel("relative world y")
        ax.set_title(f"{subject_dir.name} – Combined gaze directions")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=gaze_t_s.min(), vmax=gaze_t_s.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("time [s]")
        sns.despine(ax=ax)
        if SAVE_PLOTS:
            plt.savefig(outdir / "combined_gaze_direction_fan.png", dpi=400)
        plt.show()

