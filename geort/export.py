# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.config_utils import (
    load_json,
    parse_config_joint_limit,
    parse_config_keypoint_info,
)
from geort.utils.path import get_run_root


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''

    def __init__(self, model_path, config_path, device=None):
        model_path = Path(model_path)
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}")

        self.device = torch.device(device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"))
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(
            keypoint_joints=keypoint_info["joint"]).to(self.device)
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=True)
        checkpoint_state = checkpoint.get("state_dict", checkpoint)
        prefix = "ik_model."
        state = {
            key[len(prefix):]: value
            for key, value in checkpoint_state.items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError(
                f"No IK model weights found in checkpoint: {model_path}")
        self.model.load_state_dict(state)
        self.model.eval()
        # GeoRT will do normalization.
        self.qpos_normalizer = HandFormatter(
            joint_lower_limit, joint_upper_limit)

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids]  # extract.
        keypoints = torch.from_numpy(keypoints).unsqueeze(
            0).reshape(1, -1, 3).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            joint_normalized = self.model(keypoints)
        joint_raw = self.qpos_normalizer.unnormalize(
            joint_normalized.cpu().numpy())
        return joint_raw[0]


def load_model(tag_or_path, epoch=None, checkpoint_root=None, device=None):
    '''
        Loading API.
    '''
    candidate = Path(tag_or_path) if tag_or_path else None
    if candidate is not None and candidate.is_dir():
        resolved_dir = candidate
    else:
        if not tag_or_path:
            raise FileNotFoundError("Checkpoint tag or directory is required")
        resolved_dir = Path(
            checkpoint_root or get_run_root()) / str(tag_or_path)

    if not resolved_dir.is_dir():
        raise FileNotFoundError(
            f"Run or checkpoint directory not found: {resolved_dir}")

    checkpoint_dir = (
        resolved_dir / "checkpoints"
        if (resolved_dir / "checkpoints").is_dir()
        else resolved_dir
    )

    if epoch is None:
        model_path = checkpoint_dir / "best.ckpt"
        if not model_path.is_file():
            model_path = checkpoint_dir / "last.ckpt"
    elif epoch < 0:
        model_path = checkpoint_dir / "last.ckpt"
    else:
        model_path = checkpoint_dir / f"epoch={epoch:04d}.ckpt"
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file() and checkpoint_dir.name == "checkpoints":
        config_path = checkpoint_dir.parent / "config.json"
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path, device=device)
