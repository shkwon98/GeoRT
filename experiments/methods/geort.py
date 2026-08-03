import json
import warnings
from pathlib import Path

import torch

from experiments.schema import NamedCommand
from experiments.schema import validate_calibration
from geort.export import load_model
from geort.utils.config_utils import (
    load_json,
    parse_config_joint_limit,
    parse_config_keypoint_info,
)


_OPTIONS = {
    "device",
    "tag",
    "epoch",
    "seed",
    "val_fraction",
    "coverage_samples",
    "coverage_batch_size",
    "gesture_batch_size",
    "direction_sigma",
    "flatness_sigma",
    "w_chamfer",
    "w_curvature",
    "w_pinch",
    "w_collision",
}


class GeoRTMethod:
    def __init__(self, checkpoint_dir, robot_spec, device=None):
        self.model = load_model(checkpoint_dir, device=device)
        self.joint_names = tuple(robot_spec["joint_order"])
        limits = self.model.qpos_normalizer.joint_lower_limit
        if len(self.joint_names) != len(limits):
            raise ValueError("robot joint_order does not match the GeoRT checkpoint")

    def infer(self, frame):
        if not frame.valid:
            raise ValueError("cannot infer from an invalid canonical frame")
        return NamedCommand(
            self.joint_names,
            self.model.forward(frame.points),
            frame.timestamp,
        )


class _RolloutModel(torch.nn.Module):
    def __init__(self, ik_model, human_ids, lower, upper):
        super().__init__()
        self.ik_model = ik_model.eval().requires_grad_(False)
        self.register_buffer(
            "human_ids", torch.tensor(human_ids, dtype=torch.long)
        )
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.float32))

    def forward(self, points):
        selected = torch.index_select(points, 0, self.human_ids).unsqueeze(0)
        normalized = self.ik_model(selected)[0]
        return (normalized + 1.0) * 0.5 * (self.upper - self.lower) + self.lower


def export_torchscript(
    checkpoint_dir,
    robot_spec,
    output_path,
    experiment,
    device="cpu",
):
    checkpoint_dir = Path(checkpoint_dir)
    config = load_json(checkpoint_dir / "config.json")
    if config.get("name") != robot_spec.get("name"):
        raise ValueError("checkpoint robot does not match robot specification")
    if config.get("joint_order") != robot_spec.get("joint_order"):
        raise ValueError("checkpoint joint names do not match robot specification")
    if config.get("hand_side", robot_spec.get("hand_side")) != robot_spec.get(
        "hand_side"
    ):
        raise ValueError("checkpoint hand side does not match robot specification")

    calibration = experiment.get("calibration", {})
    scale, rotation, outward_sign = validate_calibration(
        calibration.get("scale"),
        calibration.get("rotation"),
        calibration.get("outward_sign"),
    )
    keypoints = parse_config_keypoint_info(config)
    lower, upper = parse_config_joint_limit(config)
    loaded = load_model(checkpoint_dir, device=device)
    wrapper = _RolloutModel(loaded.model, keypoints["human_id"], lower, upper).to(
        loaded.device
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(
            wrapper,
            torch.zeros(21, 3, device=loaded.device),
            strict=True,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "robot": robot_spec["name"],
        "hand_side": robot_spec["hand_side"],
        "joint_names": list(robot_spec["joint_order"]),
        "joint_lower": lower.tolist(),
        "joint_upper": upper.tolist(),
        "human_landmark_ids": keypoints["human_id"],
        "urdf_sha256": robot_spec["urdf_sha256"],
        "canonical_units": "m",
        "canonical_x": "outward palm normal",
        "canonical_y": "thumb-side palm axis",
        "canonical_z": "wrist to middle MCP",
        "mocap": experiment.get("mocap"),
        "calibration": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "outward_sign": outward_sign,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, metadata_path


def train(canonical_path, robot_spec, run_dir, options):
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown GeoRT options: {sorted(unknown)}")

    from geort.trainer import GeoRTTrainer

    run_dir = Path(run_dir)
    trainer = GeoRTTrainer(
        robot_spec,
        device=options.get("device"),
        data_dir=run_dir / "robot_data",
        checkpoint_dir=run_dir / "checkpoints",
    )
    train_options = {
        "tag": options.get("tag", "geort"),
        "epoch": int(options.get("epoch", 50)),
        "seed": int(options.get("seed", 0)),
        "val_fraction": float(options.get("val_fraction", 0.1)),
        "coverage_samples": int(options.get("coverage_samples", 20000)),
        "coverage_batch_size": int(options.get("coverage_batch_size", 2048)),
        "gesture_batch_size": int(options.get("gesture_batch_size", 2048)),
    }
    for name in (
        "direction_sigma",
        "flatness_sigma",
        "w_chamfer",
        "w_curvature",
        "w_pinch",
        "w_collision",
    ):
        if name in options:
            train_options[name] = float(options[name])
    return trainer.train(canonical_path, **train_options)
