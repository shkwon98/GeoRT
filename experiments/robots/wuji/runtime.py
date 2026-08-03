import json
from pathlib import Path

import numpy as np
import torch

from experiments.schema import NamedCommand, validate_calibration


class TorchScriptRuntime:
    def __init__(
        self,
        model_path,
        metadata_path,
        robot_spec,
        alpha=0.2,
        device="cpu",
    ):
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata not found: {metadata_path}")
        alpha = float(alpha)
        if not 0 < alpha <= 1:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.joint_names = tuple(metadata.get("joint_names", ()))
        if (
            metadata.get("robot") != robot_spec.get("name")
            or metadata.get("hand_side") != robot_spec.get("hand_side")
            or metadata.get("urdf_sha256") != robot_spec.get("urdf_sha256")
            or self.joint_names != tuple(robot_spec.get("joint_order", ()))
        ):
            raise ValueError("model metadata does not match robot specification")
        raw_lower = np.asarray(metadata.get("joint_lower"), dtype=np.float64)
        raw_upper = np.asarray(metadata.get("joint_upper"), dtype=np.float64)
        if (
            raw_lower.shape != (len(self.joint_names),)
            or raw_upper.shape != raw_lower.shape
            or not np.isfinite(raw_lower).all()
            or not np.isfinite(raw_upper).all()
            or np.any(raw_lower >= raw_upper)
        ):
            raise ValueError("model metadata has invalid joint limits")
        self.lower = raw_lower.astype(np.float32)
        self.upper = raw_upper.astype(np.float32)
        self.lower = np.where(
            self.lower.astype(np.float64) < raw_lower,
            np.nextafter(self.lower, np.float32(np.inf)),
            self.lower,
        )
        self.upper = np.where(
            self.upper.astype(np.float64) > raw_upper,
            np.nextafter(self.upper, np.float32(-np.inf)),
            self.upper,
        )
        if np.any(self.lower >= self.upper):
            raise ValueError("joint limits collapse at float32 precision")
        if metadata.get("mocap") != "webxr":
            raise ValueError("model metadata must use webxr mocap")
        calibration = metadata.get("calibration", {})
        validate_calibration(
            calibration.get("scale"),
            calibration.get("rotation"),
            calibration.get("outward_sign"),
        )

        self.metadata = metadata
        self.hand_side = metadata["hand_side"]
        self.alpha = alpha
        self.device = torch.device(device)
        self.model = torch.jit.load(
            str(model_path), map_location=self.device
        ).eval()
        self.last_command = None

    def infer(self, frame):
        if not frame.valid or frame.hand_side != self.hand_side:
            return None
        with torch.inference_mode():
            output = (
                self.model(
                    torch.from_numpy(frame.points).to(
                        self.device, dtype=torch.float32
                    )
                )
                .cpu()
                .numpy()
            )
        if output.shape != self.lower.shape or not np.isfinite(output).all():
            return None
        clipped = np.clip(output, self.lower, self.upper)
        if self.last_command is not None:
            clipped = (
                self.alpha * clipped
                + (1.0 - self.alpha) * self.last_command.qpos
            )
        command = NamedCommand(self.joint_names, clipped, frame.timestamp)
        self.last_command = command
        return command
