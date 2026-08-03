import json

import numpy as np
import pytest
import torch

from experiments.robots.wuji.runtime import TorchScriptRuntime
from experiments.schema import CanonicalFrame


class Model(torch.nn.Module):
    def forward(self, points):
        return torch.stack((points[4, 0] * 10.0, points[8, 1] * 10.0))


def _artifact(tmp_path):
    model_path = tmp_path / "model.ts"
    torch.jit.trace(Model(), torch.zeros(21, 3)).save(str(model_path))
    metadata = {
        "robot": "wuji_right",
        "hand_side": "right",
        "urdf_sha256": "a" * 64,
        "joint_names": ["right_a", "right_b"],
        "joint_lower": [-1.0, -2.0],
        "joint_upper": [1.0, 2.0],
        "mocap": "webxr",
        "calibration": {
            "scale": 1.0,
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "outward_sign": 1,
        },
    }
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(json.dumps(metadata))
    return model_path, metadata_path


def test_runtime_clips_then_filters_and_keeps_last_valid_command(tmp_path):
    model_path, metadata_path = _artifact(tmp_path)
    spec = {
        "name": "wuji_right",
        "hand_side": "right",
        "urdf_sha256": "a" * 64,
        "joint_order": ["right_a", "right_b"],
    }
    runtime = TorchScriptRuntime(model_path, metadata_path, spec, alpha=0.5)
    points = np.zeros((21, 3), dtype=np.float32)
    points[4, 0], points[8, 1] = 0.2, -0.3

    first = runtime.infer(CanonicalFrame(points, 1.0, "right"))
    np.testing.assert_array_equal(first.qpos, [1.0, -2.0])
    second = runtime.infer(
        CanonicalFrame(np.zeros((21, 3)), 1.1, "right")
    )
    np.testing.assert_array_equal(second.qpos, [0.5, -1.0])
    assert (
        runtime.infer(
            CanonicalFrame(np.zeros((21, 3)), 1.2, "right", valid=False)
        )
        is None
    )
    assert runtime.last_command is second


def test_runtime_rejects_metadata_mismatch_before_inference(tmp_path):
    model_path, metadata_path = _artifact(tmp_path)
    wrong = {
        "name": "wuji_left",
        "hand_side": "left",
        "urdf_sha256": "b" * 64,
        "joint_order": ["left_a", "left_b"],
    }
    with pytest.raises(ValueError, match="metadata"):
        TorchScriptRuntime(model_path, metadata_path, wrong)
