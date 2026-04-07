from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def transform_imu_to_world(imu_coordinates: np.ndarray, imu_quaternions_wxyz: np.ndarray) -> np.ndarray:
    """
    Transform vector(s) from IMU coordinates into world coordinates.

    imu_quaternions_wxyz: shape (N,4) in order [w, x, y, z]
    imu_coordinates:
      - shape (3,) for one vector (will be transformed for all N samples)
      - shape (N,3) for per-sample vectors
    """
    imu_to_world_matrices = R.from_quat(imu_quaternions_wxyz, scalar_first=True).as_matrix()  # (N,3,3)

    imu_coordinates = np.asarray(imu_coordinates, dtype=float)

    if imu_coordinates.ndim == 1:
        # Apply same vector to all matrices -> result (N,3)
        return imu_to_world_matrices @ imu_coordinates
    else:
        # Per-sample vectors -> result (N,3)
        return np.einsum("nij,nj->ni", imu_to_world_matrices, imu_coordinates)


def imu_heading_in_world(imu_quaternions_wxyz: np.ndarray) -> np.ndarray:
    """
    Heading vector in world coordinates.
    In the Pupil snippet, the 'neutral heading' in IMU coords is [0, 1, 0].
    """
    heading_neutral_in_imu = np.array([0.0, 1.0, 0.0])
    return transform_imu_to_world(heading_neutral_in_imu, imu_quaternions_wxyz)


def quaternion_norms(imu_quaternions_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(imu_quaternions_wxyz, dtype=float)
    return np.sqrt(np.sum(q * q, axis=1))


def interpolate_quaternions_wxyz(
    source_timestamps_ns: np.ndarray,
    source_quaternions_wxyz: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """
    Interpolate quaternions onto a new timebase using spherical interpolation.

    Timestamps are expected in nanoseconds and quaternions in [w, x, y, z] order.
    Target timestamps must lie inside the source time range.
    """
    source_timestamps_ns = np.asarray(source_timestamps_ns, dtype=np.int64)
    source_quaternions_wxyz = np.asarray(source_quaternions_wxyz, dtype=float)
    target_timestamps_ns = np.asarray(target_timestamps_ns, dtype=np.int64)

    if source_timestamps_ns.ndim != 1:
        raise ValueError("source_timestamps_ns must be a 1D array")
    if target_timestamps_ns.ndim != 1:
        raise ValueError("target_timestamps_ns must be a 1D array")
    if source_quaternions_wxyz.shape != (len(source_timestamps_ns), 4):
        raise ValueError("source_quaternions_wxyz must have shape (N, 4)")
    if len(source_timestamps_ns) < 2:
        raise ValueError("Need at least two source quaternions for interpolation")

    keep = np.concatenate(([True], np.diff(source_timestamps_ns) > 0))
    source_timestamps_ns = source_timestamps_ns[keep]
    source_quaternions_wxyz = source_quaternions_wxyz[keep]

    if len(source_timestamps_ns) < 2:
        raise ValueError("Need at least two unique source timestamps for interpolation")
    if np.any(target_timestamps_ns < source_timestamps_ns[0]) or np.any(target_timestamps_ns > source_timestamps_ns[-1]):
        raise ValueError("target_timestamps_ns must lie within the source timestamp range")

    rotations = R.from_quat(source_quaternions_wxyz, scalar_first=True)
    slerp = Slerp(source_timestamps_ns.astype(np.float64), rotations)
    return slerp(target_timestamps_ns.astype(np.float64)).as_quat(scalar_first=True)
