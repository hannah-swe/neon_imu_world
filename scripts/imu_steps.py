from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from neon_imu.loaders import load_imu_csv
from neon_imu.transforms import quaternion_norms, transform_imu_to_world
from neon_imu.plot_config import setup_plot_style
setup_plot_style()

"""This script:
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
SUBJECT_GLOB = "sub-*"
IMU_FILENAME = "imu.csv"
SHOW_PLOTS = True


def wrap_deg(a: np.ndarray) -> np.ndarray:
    # Wrap angles in degrees to [-180, 180]
    return (a + 180) % 360 - 180


# 1) Discover subject folders
subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

# 2) Iterate subjects (linear)
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

    # 2.4) Quaternion norm sanity check; quaternion norm should be 1 (length = 1)
    norms = quaternion_norms(q_wxyz)
    print(f"  Quaternion norm min/mean/max: {norms.min():.6f} / {norms.mean():.6f} / {norms.max():.6f}")
    bad_idx = np.where(np.abs(norms - 1.0) > 1e-2)[0]  # tolerance 0.01
    print(f"  Samples with |norm-1| > 0.01: {len(bad_idx)}")

    # 2.5) Build time axis in seconds (relative, start (0 s) at first timepoint)
    t_ns = df["timestamp [ns]"].to_numpy(dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9

    # 2.6) Pupil labs yaw (rotation over vertical axis;
    # same information as in quaternion but readable for humans in degrees)
    csv_yaw = df["yaw [deg]"].to_numpy(dtype=float)
    csv_yaw_wrapped = wrap_deg(csv_yaw) # wrap it from -180 to 180 degrees

    # Heading definition that matches CSV yaw:
    # axis = -Y in IMU coords, rotated into world,
    # yaw = atan2(world_y, world_x) + 90°, wrapped to [-180, 180]
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

    # Plots:
    if SHOW_PLOTS:
        # A) CSV yaw vs derived yaw
        plt.figure()
        sns.lineplot(x=t_s, y=csv_yaw_wrapped, label="CSV yaw [deg]", linewidth=7, alpha=0.7)
        sns.lineplot(x=t_s, y=heading_angle_deg, label="Derived yaw from -Y (+90°) [deg]", linewidth=1.75)
        plt.xlabel("time [s]")
        plt.ylabel("deg (wrapped to [-180, 180])")
        plt.title(f"{subject_dir.name} – CSV yaw vs derived yaw")
        plt.legend()
        sns.despine()
        plt.show()

        # B) Error plot
        plt.figure()
        sns.lineplot(x=t_s, y=err, linewidth=1.5)
        plt.xlabel("time [s]")
        plt.ylabel("deg")
        plt.title(f"{subject_dir.name} – Error (derived - CSV), RMS={rms:.8f}°")
        sns.despine()
        plt.show()

        # C) Heading vector components in world
        plt.figure()
        sns.lineplot(x=t_s, y=heading_world_unit[:, 0], label="heading x", linewidth=2.5)
        sns.lineplot(x=t_s, y=heading_world_unit[:, 1], label="heading y", linewidth=2.5)
        sns.lineplot(x=t_s, y=heading_world_unit[:, 2], label="heading z", linewidth=2.5)
        plt.xlabel("time [s]")
        plt.ylabel("unit heading component")
        plt.title(f"{subject_dir.name} – Heading vector (world)")
        plt.legend()
        sns.despine()
        plt.show()

        # D) Acceleration magnitude in world (g)
        plt.figure()
        sns.lineplot(x=t_s, y=acc_world_mag_g, linewidth=1.5)
        plt.axhline(1.0, linestyle="--", color="grey")
        plt.xlabel("time [s]")
        plt.ylabel("||acc|| [g]")
        plt.title(f"{subject_dir.name} – Acc magnitude (world) [g]")
        sns.despine()
        plt.show()

        # E) Z-component of heading over time (pitch tendency)
        plt.figure()
        sns.lineplot(x=t_s, y=heading_world_unit[:, 2], linewidth=2.5)
        plt.xlabel("time [s]")
        plt.ylabel("heading z-component")
        plt.title(f"{subject_dir.name} – Heading Z component (vertical tilt)")
        plt.axhline(0, linestyle="--", color="grey")
        plt.show()

        # F) Heading directions projected onto horizontal plane
        plt.figure()
        xy = heading_world_unit[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
        xy_unit = xy / np.maximum(xy_norm, 1e-12)
        cmap = sns.color_palette("crest_r", as_cmap=True)
        t_norm = (t_s - t_s.min()) / (t_s.max() - t_s.min())
        for i, (x_val, y_val) in enumerate(xy_unit):
            plt.plot(
                [0, x_val],
                [0, y_val],
                color=cmap(t_norm[i]),
                alpha=0.1
            )
        theta = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(theta), np.sin(theta), linestyle="-", color="grey", alpha=0.8)
        plt.xlabel("world x")
        plt.ylabel("world y")
        plt.title(f"{subject_dir.name} – Heading direction")
        plt.gca().set_aspect("equal", adjustable="box")
        # FIXED COLORBAR
        ax = plt.gca()
        norm = mcolors.Normalize(vmin=t_s.min(), vmax=t_s.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("time [s]")
        plt.show()
