# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.spatial.transform import Rotation


MANUS_LINK_OFFSETS = np.array([
    [0.0, 0.0, 0.0],
    [0.0250, 0.0000, 0.0050],
    [0.0000, 0.0000, 0.0390],
    [0.0000, 0.0000, 0.0330],
    [0.0000, 0.0000, 0.0210],
    [0.0170, 0.0000, 0.0870],
    [0.0000, 0.0000, 0.0260],
    [0.0000, 0.0000, 0.0220],
    [0.0000, 0.0000, 0.0200],
    [0.0000, 0.0000, 0.0920],
    [0.0000, 0.0000, 0.0260],
    [0.0000, 0.0000, 0.0260],
    [0.0000, 0.0000, 0.0220],
    [-0.0170, 0.0000, 0.0840],
    [0.0000, 0.0000, 0.0210],
    [0.0000, 0.0000, 0.0210],
    [0.0000, 0.0000, 0.0200],
    [-0.0340, 0.0000, 0.0720],
    [0.0000, 0.0000, 0.0210],
    [0.0000, 0.0000, 0.0210],
    [0.0000, 0.0000, 0.0200],
])

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def _solve_keypoints(orientations):
    keypoints = {}
    for chain in FINGER_CHAINS:
        transform = np.eye(4)
        for index in chain:
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(
                orientations[index]).as_matrix()
            local[:3, 3] = MANUS_LINK_OFFSETS[index]
            transform = transform @ local
            keypoints.setdefault(index, transform[:3, 3].copy())
    return np.array([keypoints[index] for index in range(21)])


def _hand_to_canonical(points):
    z_axis = points[9] - points[0]
    z_axis /= np.linalg.norm(z_axis)
    y_axis_aux = points[5] - points[13]
    y_axis_aux /= np.linalg.norm(y_axis_aux)

    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = points[0]
    homogeneous = np.column_stack((points, np.ones(21)))
    return homogeneous @ np.linalg.inv(transform).T


def manus_keypoints(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.shape != (21, 4):
        raise ValueError(
            f"Expected Manus quaternions with shape (21, 4), got {quaternions.shape}"
        )
    if not np.isfinite(quaternions).all():
        raise ValueError("Manus quaternions must contain only finite values")

    points = _hand_to_canonical(_solve_keypoints(quaternions))[:, :3]
    return points.astype(np.float32)
