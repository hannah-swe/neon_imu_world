from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


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