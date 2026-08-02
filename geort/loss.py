# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 


def collision_free_loss(collision_logits):
    return torch.nn.functional.softplus(collision_logits).mean()


def pinch_correspondence_loss(human_keypoints, robot_keypoints, threshold=0.015):
    if human_keypoints.ndim != 3 or robot_keypoints.ndim != 3:
        raise ValueError("human_keypoints and robot_keypoints must have shape [B, F, 3]")
    if human_keypoints.shape != robot_keypoints.shape or human_keypoints.shape[-1] != 3:
        raise ValueError("human_keypoints and robot_keypoints must have matching shape [B, F, 3]")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    loss = robot_keypoints.new_zeros(())
    for i in range(human_keypoints.size(1)):
        for j in range(i + 1, human_keypoints.size(1)):
            human_distance = torch.norm(human_keypoints[:, i] - human_keypoints[:, j], dim=-1)
            robot_distance_squared = ((robot_keypoints[:, i] - robot_keypoints[:, j]) ** 2).sum(dim=-1)
            loss += ((human_distance < threshold).to(robot_distance_squared.dtype) * robot_distance_squared).mean()
    return loss


def chamfer_distance(input_points, target_points):
    """
    Args:
    - input_points (torch.Tensor): Input point cloud tensor of shape [B, N, 3].
    - target_points (torch.Tensor): Target point cloud tensor of shape [B, M, 3].
    
    Returns:
    - chamfer_dist (torch.Tensor): Chamfer distance.
    """
    B, N, _ = input_points.size()
    _, M, _ = target_points.size()
    
    input_points = input_points.clone()
    target_points = target_points.clone()
    input_points[..., 1] = input_points[..., 1] 
    target_points[..., 1] = target_points[..., 1]

    input_points = input_points.unsqueeze(2)    # [B, N, 1, 3]
    target_points = target_points.unsqueeze(1)  # [B, 1, M, 3]
    
    input_points_repeat = input_points.repeat(1, 1, M, 1)    # [B, N, M, 3]
    target_points_repeat = target_points.repeat(1, N, 1, 1)  # [B, N, M, 3]
    

    dist_matrix = torch.sum((input_points_repeat - target_points_repeat)**2, dim=-1)  # [B, N, M]
    
    min_dist_a, _ = torch.min(dist_matrix, dim=2)  # [B, N]
    min_dist_b, _ = torch.min(dist_matrix, dim=1)  # [B, M]
    
    chamfer_dist = torch.mean(min_dist_a, dim=1) + torch.mean(min_dist_b, dim=1)
    
    return chamfer_dist.mean()
