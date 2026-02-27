from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neon_imu.loaders import load_imu_csv
from neon_imu.transforms import imu_heading_in_world, quaternion_norms, transform_imu_to_world


imu_path = Path("data/raw/imu.csv")
df = load_imu_csv(imu_path)

# ---- Extract quaternions in correct order: [w, x, y, z]
q_wxyz = df[["quaternion w", "quaternion x", "quaternion y", "quaternion z"]].to_numpy(dtype=float)

# ---- 1) Quaternion norm check (should be ~1.0)
norms = quaternion_norms(q_wxyz)
print("Quaternion norm stats:")
print(f"  min:  {norms.min():.6f}")
print(f"  mean: {norms.mean():.6f}")
print(f"  max:  {norms.max():.6f}")

# flag if something looks off
bad = np.where(np.abs(norms - 1.0) > 1e-2)[0]  # 0.01 tolerance
print(f"Samples with |norm-1| > 0.01: {len(bad)}")

# ---- 2) Heading vector in world coords
heading_world = imu_heading_in_world(q_wxyz)  # shape (N,3)

# Optional: normalize heading (usually already ~1)
heading_norm = np.linalg.norm(heading_world, axis=1, keepdims=True)
heading_world_unit = heading_world / np.maximum(heading_norm, 1e-12)

# ---- 3) Build a time axis in seconds (relative)
t_ns = df["timestamp [ns]"].to_numpy(dtype=np.int64)
t_s = (t_ns - t_ns[0]) / 1e9

# ---- Plot heading components
plt.figure()
plt.plot(t_s, heading_world_unit[:, 0], label="heading_world x")
plt.plot(t_s, heading_world_unit[:, 1], label="heading_world y")
plt.plot(t_s, heading_world_unit[:, 2], label="heading_world z")
plt.xlabel("time [s]")
plt.ylabel("unit heading component")
plt.title("IMU heading vector in world coordinates")
plt.legend()
plt.tight_layout()
plt.show()

# ---- 4) OPTIONAL: accelerations -> world
# Note: acceleration columns are in g. Convert to m/s^2 if you want:
# 1 g = 9.80665 m/s^2
acc_g = df[["acceleration x [g]", "acceleration y [g]", "acceleration z [g]"]].to_numpy(dtype=float)
acc_world_g = transform_imu_to_world(acc_g, q_wxyz)

# quick plot of norm of acceleration in g (should be ~1 when only gravity)
acc_norm = np.linalg.norm(acc_world_g, axis=1)

plt.figure()
plt.plot(t_s, acc_norm)
plt.xlabel("time [s]")
plt.ylabel("||acc|| [g]")
plt.title("Acceleration magnitude (world), in g")
plt.tight_layout()
plt.show()
