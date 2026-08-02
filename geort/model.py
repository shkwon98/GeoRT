# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def get_finger_fk(n_joint=4, hidden=128):
    return nn.Sequential(
        nn.Linear(n_joint, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, 3)
    ) 

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

class FKModel(nn.Module):
    def __init__(self, keypoint_joints):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: For allegro, [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        num_fingers = len(keypoint_joints)
        
        self.nets = []
        self.n_total_joint = 0

        for joint in keypoint_joints:
            net = get_finger_fk(n_joint=len(joint))
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
    def __init__(self, keypoint_joints, num_joints=None, hidden_dim=128):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        if not keypoint_joints or any(not joint for joint in keypoint_joints):
            raise ValueError("keypoint_joints must contain at least one joint per finger")

        self.keypoint_joints = [tuple(joint) for joint in keypoint_joints]
        joint_ids = [joint_id for joint in self.keypoint_joints for joint_id in joint]
        if min(joint_ids) < 0:
            raise ValueError("joint indices must be non-negative")
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("a joint may only be assigned to one FingerIK")

        inferred_num_joints = max(joint_ids) + 1
        self.n_total_joint = inferred_num_joints if num_joints is None else num_joints
        if self.n_total_joint < inferred_num_joints:
            raise ValueError("num_joints must include every configured joint index")

        self.num_fingers = len(self.keypoint_joints)
        # Keep the registered name for existing IK checkpoint compatibility.
        self.nets = nn.ModuleList([
            FingerIK(num_joints=len(joint), hidden_dim=hidden_dim)
            for joint in self.keypoint_joints
        ])

    @property
    def finger_models(self):
        return self.nets

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
