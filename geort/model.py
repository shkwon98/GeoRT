# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def _validate_joint_groups(keypoint_joints):
    groups = [list(group) for group in keypoint_joints]
    flat = [index for group in groups for index in group]
    if (
        not groups
        or any(not group for group in groups)
        or any(not isinstance(index, int) or index < 0 for index in flat)
        or sorted(flat) != list(range(len(flat)))
    ):
        raise ValueError(
            "joint groups must be non-empty, disjoint, and cover indices 0..DOF-1"
        )
    return groups


class FingerFK(nn.Sequential):
    """Maps one finger's joint values to its fingertip coordinate."""

    def __init__(self, num_joints=4, hidden_dim=128):
        if num_joints < 1:
            raise ValueError("num_joints must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        super().__init__(
            nn.Linear(num_joints, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 3),
        )
        self.num_joints = num_joints

    def forward(self, joints):
        if joints.ndim != 2 or joints.shape[1] != self.num_joints:
            raise ValueError(
                f"Expected joints with shape [B, {self.num_joints}], "
                f"got {tuple(joints.shape)}"
            )
        return super().forward(joints)


class FingerIK(nn.Sequential):
    """Maps one fingertip coordinate to the joints of one finger."""

    def __init__(self, num_joints=4, hidden_dim=128):
        if num_joints < 1:
            raise ValueError("num_joints must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        super().__init__(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_joints),
            nn.Tanh(),  # Normalize.
        )
        self.num_joints = num_joints

    def forward(self, fingertip):
        if fingertip.ndim != 2 or fingertip.shape[1] != 3:
            raise ValueError(
                "Expected fingertip with shape [B, 3], "
                f"got {tuple(fingertip.shape)}"
            )
        return super().forward(fingertip)


class FKModel(nn.Module):
    def __init__(self, keypoint_joints, hidden_dim=128):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: For allegro, [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        keypoint_joints = _validate_joint_groups(keypoint_joints)

        self.nets = []
        self.n_total_joint = 0

        for joint in keypoint_joints:
            net = FingerFK(num_joints=len(joint), hidden_dim=hidden_dim)
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)

        self.keypoint_joints = keypoint_joints

    def forward(self, joint):
        # x: [B, DOF], joint values. normalized to [-1, 1].
        # out:   [B, N, 3], sequence of keypoint.
        keypoints = []
        for i, net in enumerate(self.nets):
            joint_ids = self.keypoint_joints[i]
            keypoint = net(joint[:, joint_ids])
            keypoints.append(keypoint)

        return torch.stack(keypoints, dim=1)


class IKModel(nn.Module):
    def __init__(self, keypoint_joints, hidden_dim=128):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        keypoint_joints = _validate_joint_groups(keypoint_joints)
        self.n_total_joint = sum(len(joint) for joint in keypoint_joints)
        self.num_fingers = len(keypoint_joints)
        # Keep the registered name for existing IK checkpoint compatibility.
        self.nets = nn.ModuleList([
            FingerIK(num_joints=len(joint), hidden_dim=hidden_dim)
            for joint in keypoint_joints
        ])
        self.keypoint_joints = keypoint_joints

    def forward(self, x):
        # x:   [B, N, 3], sequence of keypoint.
        # out: [B, DOF], joint values. normalized to [-1, 1].
        if x.ndim != 3 or x.shape[1] != self.num_fingers or x.shape[2] != 3:
            raise ValueError(
                f"Expected keypoints with shape [B, {self.num_fingers}, 3], "
                f"got {tuple(x.shape)}"
            )

        out = x.new_zeros((x.size(0), self.n_total_joint))
        for i, net in enumerate(self.nets):
            joint = net(x[:, i])
            out[:, self.keypoint_joints[i]] = joint
        return out


class CollisionClassifier(nn.Module):
    """Predicts a collision logit from normalized joint values."""

    def __init__(self, num_joints, hidden_dim=128):
        super().__init__()
        if num_joints < 1:
            raise ValueError("num_joints must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.num_joints = num_joints
        self.net = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joints):
        if joints.ndim != 2 or joints.shape[1] != self.num_joints:
            raise ValueError(
                f"Expected normalized joints with shape [B, {self.num_joints}], "
                f"got {tuple(joints.shape)}"
            )
        if not torch.is_floating_point(joints) or not torch.isfinite(joints).all() or (joints.abs() > 1).any():
            raise ValueError("Expected joints normalized to [-1, 1]")
        return self.net(joints).squeeze(-1)
