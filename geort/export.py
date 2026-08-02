# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.path import get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


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
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
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
        checkpoint_dir = candidate
    else:
        if not tag_or_path:
            raise FileNotFoundError("Checkpoint tag or directory is required")
        checkpoint_dir = Path(
            checkpoint_root or get_checkpoint_root()) / str(tag_or_path)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}")

    model_path = checkpoint_dir / \
        ("last.pth" if epoch is None or epoch < 0 else f"epoch_{epoch}.pth")
    config_path = checkpoint_dir / "config.json"
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path, device=device)


if __name__ == '__main__':
    # load the model in one line.
    load_model("allegro_last")
