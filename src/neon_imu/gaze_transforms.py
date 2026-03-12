from __future__ import annotations
import numpy as np
from neon_imu.transforms import transform_imu_to_world


def spherical_to_cartesian_scene(elevations_deg: np.ndarray, azimuths_deg: np.ndarray) -> np.ndarray:
    """
    Convert Neon's spherical representation of 3D gaze (elevation/azimuth in deg)
    into Cartesian direction vectors in SCENE coordinates.

    Output shape: (N, 3)
    """
    elevations_rad = np.deg2rad(elevations_deg)
    azimuths_rad = np.deg2rad(azimuths_deg)

    # Convert to a more traditional spherical coordinate convention (as in docs)
    elevations_rad += np.pi / 2
    azimuths_rad *= -1.0
    azimuths_rad += np.pi / 2

    return np.array(
        [
            np.sin(elevations_rad) * np.cos(azimuths_rad),
            np.cos(elevations_rad),
            np.sin(elevations_rad) * np.sin(azimuths_rad),
        ]
    ).T


def cartesian_to_spherical_world(world_points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert points/vectors in 3D Cartesian WORLD coordinates to spherical coordinates.

    Returns:
      elevation_deg, azimuth_deg  (both in degrees)
    """
    x = world_points_3d[:, 0]
    y = world_points_3d[:, 1]
    z = world_points_3d[:, 2]

    radii = np.sqrt(x**2 + y**2 + z**2)
    radii = np.maximum(radii, 1e-12)

    elevation = -(np.arccos(z / radii) - np.pi / 2)
    azimuth = np.arctan2(y, x) - np.pi / 2

    # Wrap azimuth to [-pi, pi]
    azimuth[azimuth < -np.pi] += 2 * np.pi
    azimuth[azimuth > np.pi] -= 2 * np.pi

    return np.rad2deg(elevation), np.rad2deg(azimuth)


def transform_scene_dirs_to_world(
    cart_dirs_in_scene: np.ndarray,
    imu_quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D DIRECTION vectors from SCENE coordinates to WORLD coordinates.

    Directions only -> no translation. We apply:
      1) fixed rotation: scene -> imu
      2) per-sample rotation: imu -> world (quaternions)

    Output shape: (N,3)
    """
    imu_scene_rotation_diff = np.deg2rad(-90 - 12)
    scene_to_imu = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(imu_scene_rotation_diff), -np.sin(imu_scene_rotation_diff)],
            [0.0, np.sin(imu_scene_rotation_diff),  np.cos(imu_scene_rotation_diff)],
        ]
    )

    dirs_in_imu = (scene_to_imu @ cart_dirs_in_scene.T).T
    return transform_imu_to_world(dirs_in_imu, imu_quaternions_wxyz)