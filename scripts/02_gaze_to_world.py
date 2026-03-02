from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from neon_imu.loaders import load_imu_csv
from neon_imu.plot_config import setup_plot_style
from neon_imu.gaze_transforms import (
    spherical_to_cartesian_scene,
    cartesian_to_spherical_world,
    transform_scene_dirs_to_world,
)
setup_plot_style()


def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180) % 360 - 180


# ----------------------------
# SETTINGS
# ----------------------------
RAW_ROOT = Path("data/raw")
SUBJECT_GLOB = "sub-*"
IMU_FILENAME = "imu.csv"
GAZE_FILENAME = "gaze.csv"   # <- falls deine Datei anders heißt, hier ändern
SHOW_PLOTS = True
# ----------------------------

subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

for subject_dir in subject_dirs:
    print("\n" + "=" * 80)
    print(f"Processing {subject_dir.name}")
    print("=" * 80)

    imu_path = subject_dir / IMU_FILENAME
    gaze_path = subject_dir / GAZE_FILENAME

    if not imu_path.exists():
        print(f"  SKIP: IMU file not found: {imu_path}")
        continue

    if not gaze_path.exists():
        print(f"  SKIP: Gaze file not found: {gaze_path}")
        continue

    # ----------------------------
    # Load IMU
    # ----------------------------
    imu_df = load_imu_csv(imu_path).sort_values("timestamp [ns]").reset_index(drop=True)
    imu_t_ns = imu_df["timestamp [ns]"].to_numpy(np.int64)
    imu_q_wxyz = imu_df[["quaternion w", "quaternion x", "quaternion y", "quaternion z"]].to_numpy(float)

    # ----------------------------
    # Load Gaze
    # ----------------------------
    gaze_df = pd.read_csv(gaze_path).sort_values("timestamp [ns]").reset_index(drop=True)
    gaze_t_ns = gaze_df["timestamp [ns]"].to_numpy(np.int64)
    gaze_t_s = (gaze_t_ns - gaze_t_ns[0]) / 1e9

    gaze_az = gaze_df["azimuth [deg]"].to_numpy(float)
    gaze_el = gaze_df["elevation [deg]"].to_numpy(float)

    # ----------------------------
    # Align gaze samples to nearest IMU quaternion by timestamp
    # ----------------------------
    imu_idx = np.searchsorted(imu_t_ns, gaze_t_ns, side="left")
    imu_idx = np.clip(imu_idx, 0, len(imu_t_ns) - 1)

    dt_ms = np.abs(imu_t_ns[imu_idx] - gaze_t_ns) / 1e6
    print(f"  Gaze->IMU alignment dt [ms]: median={np.median(dt_ms):.3f}, max={np.max(dt_ms):.3f}")

    gaze_q_wxyz = imu_q_wxyz[imu_idx]

    # ----------------------------
    # Gaze: Scene spherical -> Scene Cartesian direction vectors
    # ----------------------------
    gaze_scene_dirs = spherical_to_cartesian_scene(gaze_el, gaze_az)  # (N,3)

    # ----------------------------
    # Scene -> World (directions only, no translation)
    # ----------------------------
    gaze_world_dirs = transform_scene_dirs_to_world(gaze_scene_dirs, gaze_q_wxyz)  # (N,3)

    # Optional: World Cartesian -> World spherical (intuitive)
    gaze_world_el, gaze_world_az = cartesian_to_spherical_world(gaze_world_dirs)
    gaze_world_az = wrap_deg(gaze_world_az)

    # ----------------------------
    # PLOTS (per subject)
    # ----------------------------
    if SHOW_PLOTS:
        # A) World elevation over time
        plt.figure()
        sns.lineplot(x=gaze_t_s, y=gaze_world_el, label="world elevation [deg]")
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg")
        plt.title(f"{subject_dir.name} – Gaze elevation in WORLD")
        plt.show()

        # B) World azimuth over time
        plt.figure()
        sns.lineplot(x=gaze_t_s, y=gaze_world_az, label="world azimuth [deg]")
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg (wrapped to [-180, 180])")
        plt.title(f"{subject_dir.name} – Gaze azimuth in WORLD")
        plt.show()

        # C) Direction fan plot in horizontal plane, time-colored (crest)
        plt.figure()

        xy = gaze_world_dirs[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
        xy_unit = xy / np.maximum(xy_norm, 1e-12)

        # subsample so plot doesn't become too dense
        step = max(1, len(xy_unit) // 2000)

        cmap = sns.color_palette("crest_r", as_cmap=True)
        t_norm = (gaze_t_s - gaze_t_s.min()) / (gaze_t_s.max() - gaze_t_s.min() + 1e-12)

        for i in range(0, len(xy_unit), step):
            plt.plot([0, xy_unit[i, 0]], [0, xy_unit[i, 1]], color=cmap(t_norm[i]), alpha=0.1)

        theta = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(theta), np.sin(theta), color="grey", alpha=0.5)

        plt.xlabel("world x")
        plt.ylabel("world y")
        plt.title(f"{subject_dir.name} – Gaze direction fan (WORLD)")
        plt.gca().set_aspect("equal", adjustable="box")

        # colorbar with real time [s]
        ax = plt.gca()
        norm = mcolors.Normalize(vmin=gaze_t_s.min(), vmax=gaze_t_s.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("time [s]")

        plt.show()
