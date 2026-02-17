"""Image registration utilities for lesion matching across timepoints.

Implements ADR-010: rigid registration using centre-of-mass alignment and
mutual-information-based optimisation via ``scipy``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_volumes(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    moving_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Rigid registration returning a 4x4 transformation matrix.

    The pipeline:
      1. Centre-of-mass alignment (translation initialisation).
      2. Optimise translation + rotation by maximising normalised mutual
         information between the fixed and the transformed moving volume.

    Parameters
    ----------
    fixed:
        Reference 3-D image volume (D, H, W).
    moving:
        Moving 3-D image volume to align to *fixed*.
    fixed_spacing:
        Physical voxel spacing (z, y, x) in mm for *fixed*.
    moving_spacing:
        Physical voxel spacing (z, y, x) in mm for *moving*.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix mapping points in the
        *moving* coordinate system to the *fixed* coordinate system.
    """
    # Validate inputs
    if fixed.ndim != 3 or moving.ndim != 3:
        raise ValueError("Both volumes must be 3-D arrays")
    if fixed.size == 0 or moving.size == 0:
        return np.eye(4, dtype=np.float64)

    fs = np.asarray(fixed_spacing, dtype=np.float64)
    ms = np.asarray(moving_spacing, dtype=np.float64)

    # Step 1: centre-of-mass alignment (physical coordinates)
    com_fixed = np.asarray(
        ndimage.center_of_mass(np.abs(fixed)), dtype=np.float64
    ) * fs
    com_moving = np.asarray(
        ndimage.center_of_mass(np.abs(moving)), dtype=np.float64
    ) * ms

    translation_init = com_fixed - com_moving

    # Step 2: optimise rigid transform (translation + rotation)
    # Parameter vector: [tx, ty, tz, rx, ry, rz]  (angles in radians)
    x0 = np.array([
        translation_init[0],
        translation_init[1],
        translation_init[2],
        0.0, 0.0, 0.0,
    ], dtype=np.float64)

    # Down-sample for speed if volumes are large
    fixed_ds, moving_ds = _downsample_pair(fixed, moving, max_dim=64)

    def cost(params: np.ndarray) -> float:
        """Negative mutual information (to minimise)."""
        mat = _params_to_matrix(params)
        transformed = _apply_affine_volume(moving_ds, mat, fixed_ds.shape)
        mi = compute_mutual_information(fixed_ds, transformed, bins=32)
        return -mi

    result = optimize.minimize(
        cost,
        x0,
        method="Powell",
        options={"maxiter": 200, "ftol": 1e-6},
    )

    return _params_to_matrix(result.x)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an Nx3 array of points.

    Parameters
    ----------
    points:
        Array of shape ``(N, 3)`` with 3-D coordinates.
    transform:
        4x4 transformation matrix.

    Returns
    -------
    np.ndarray
        Transformed points with shape ``(N, 3)``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 array, got shape {points.shape}")

    n = points.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])  # Nx4
    transformed = (transform @ homogeneous.T).T  # Nx4
    return transformed[:, :3]


def compute_mutual_information(
    image1: np.ndarray,
    image2: np.ndarray,
    bins: int = 64,
) -> float:
    """Normalised mutual information between two images.

    Uses the formula ``NMI = (H(A) + H(B)) / H(A, B)`` where *H* denotes
    Shannon entropy.

    Parameters
    ----------
    image1, image2:
        Arrays of the same shape.  They are flattened internally.
    bins:
        Number of histogram bins.

    Returns
    -------
    float
        Normalised mutual information.  Returns 0.0 if inputs are
        degenerate (e.g. constant).
    """
    a = np.asarray(image1, dtype=np.float64).ravel()
    b = np.asarray(image2, dtype=np.float64).ravel()

    if a.size == 0 or b.size == 0:
        return 0.0

    # Use the shorter array length when sizes differ
    min_len = min(a.size, b.size)
    a = a[:min_len]
    b = b[:min_len]

    # Joint histogram
    joint_hist, _, _ = np.histogram2d(a, b, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()

    # Marginals
    pa = joint_prob.sum(axis=1)
    pb = joint_prob.sum(axis=0)

    # Entropies (avoid log(0))
    ha = _entropy(pa)
    hb = _entropy(pb)
    hab = _entropy(joint_prob.ravel())

    if hab == 0.0:
        return 0.0
    return (ha + hb) / hab


def align_by_center_of_mass(
    fixed_centroids: np.ndarray,
    moving_centroids: np.ndarray,
) -> np.ndarray:
    """Compute a translation-only 4x4 transform aligning centroid clouds.

    Parameters
    ----------
    fixed_centroids:
        Array of shape ``(M, 3)`` with reference centroid positions.
    moving_centroids:
        Array of shape ``(N, 3)`` with moving centroid positions.

    Returns
    -------
    np.ndarray
        4x4 translation-only matrix that shifts the centre of mass of
        *moving_centroids* to that of *fixed_centroids*.
    """
    fixed_centroids = np.atleast_2d(np.asarray(fixed_centroids, dtype=np.float64))
    moving_centroids = np.atleast_2d(np.asarray(moving_centroids, dtype=np.float64))

    if fixed_centroids.size == 0 or moving_centroids.size == 0:
        return np.eye(4, dtype=np.float64)

    com_fixed = fixed_centroids.mean(axis=0)
    com_moving = moving_centroids.mean(axis=0)
    translation = com_fixed - com_moving

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = translation
    return mat


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _params_to_matrix(params: np.ndarray) -> np.ndarray:
    """Convert 6-DOF parameter vector to 4x4 homogeneous matrix.

    Parameters are ``[tx, ty, tz, rx, ry, rz]`` where rotations are in
    radians about the z, y, x axes respectively (extrinsic Euler angles).
    """
    tx, ty, tz, rx, ry, rz = params

    # Rotation matrices for each axis
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([
        [1, 0,   0],
        [0, cx, -sx],
        [0, sx,  cx],
    ], dtype=np.float64)

    rot_y = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ], dtype=np.float64)

    rot_z = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1],
    ], dtype=np.float64)

    rotation = rot_z @ rot_y @ rot_x

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = [tx, ty, tz]
    return mat


def _apply_affine_volume(
    volume: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    """Apply a 4x4 affine to a 3-D volume using ``scipy.ndimage``.

    Returns the transformed volume resampled onto the *output_shape* grid.
    """
    # ndimage.affine_transform expects the *inverse* mapping
    inv = np.linalg.inv(matrix)
    return ndimage.affine_transform(
        volume.astype(np.float64),
        inv[:3, :3],
        offset=inv[:3, 3],
        output_shape=output_shape,
        order=1,  # bilinear interpolation
        mode="constant",
        cval=0.0,
    )


def _downsample_pair(
    vol1: np.ndarray,
    vol2: np.ndarray,
    max_dim: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically downsample both volumes so no dimension exceeds *max_dim*."""
    factor = max(1, max(max(vol1.shape), max(vol2.shape)) // max_dim)
    if factor <= 1:
        return vol1, vol2

    slices = tuple(slice(None, None, factor) for _ in range(3))
    return vol1[slices], vol2[slices]


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats for a probability distribution."""
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return -float(np.sum(p * np.log(p)))
