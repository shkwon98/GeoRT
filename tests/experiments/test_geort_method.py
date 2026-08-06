import json

import numpy as np
import pytest
import torch

from experiments.methods.geort import GeoRTMethod, export_torchscript, train
from experiments.schema import CanonicalFrame
from geort.model import IKModel
from geort.utils.config_utils import save_json


class Model:
    def forward(self, points):
        assert points.shape == (21, 3)
        return np.array([0.2, -0.1], dtype=np.float32)


def test_geort_method_returns_robot_named_command():
    method = GeoRTMethod.__new__(GeoRTMethod)
    method.model = Model()
    method.joint_names = ("joint_b", "joint_a")
    frame = CanonicalFrame(np.zeros((21, 3)), 3.5, "right")

    command = method.infer(frame)

    assert command.joint_names == ("joint_b", "joint_a")
    np.testing.assert_array_equal(
        command.qpos, np.array([0.2, -0.1], dtype=np.float32)
    )
    assert command.timestamp == 3.5


def test_train_forwards_robot_and_size_options(tmp_path, monkeypatch):
    import geort.trainer as trainer_module

    calls = {}

    class Trainer:
        def __init__(self, config, **kwargs):
            calls["config"] = config
            calls["constructor"] = kwargs

        def train(self, path, **kwargs):
            calls["path"] = path
            calls["train"] = kwargs
            return tmp_path / "checkpoint"

    monkeypatch.setattr(trainer_module, "GeoRTTrainer", Trainer)
    canonical = tmp_path / "canonical.npy"
    np.save(canonical, np.zeros((2, 21, 3), dtype=np.float32))
    spec = {"name": "wuji_right", "joint_order": ["a"]}

    result = train(
        canonical,
        spec,
        tmp_path,
        {
            "coverage_samples": 64,
            "coverage_batch_size": 8,
            "gesture_batch_size": 4,
            "save_every": 7,
            "epoch": 1,
        },
    )

    assert result == tmp_path / "checkpoint"
    assert calls["config"] is spec
    assert calls["train"]["coverage_samples"] == 64
    assert calls["train"]["coverage_batch_size"] == 8
    assert calls["train"]["gesture_batch_size"] == 4
    assert calls["train"]["save_every"] == 7


def test_train_rejects_unknown_options(tmp_path):
    with pytest.raises(ValueError, match="unknown GeoRT options"):
        train(tmp_path / "canonical.npy", {}, tmp_path, {"epoh": 1})


def _write_checkpoint(tmp_path):
    checkpoint = tmp_path / "two_finger"
    checkpoint.mkdir()
    config = {
        "name": "two_finger",
        "joint_order": ["joint_0", "joint_1"],
        "fingertip_link": [
            {
                "name": "thumb",
                "link": "finger_0",
                "center_offset": [0, 0, 0],
                "human_hand_id": 4,
                "joint": ["joint_0"],
            },
            {
                "name": "index",
                "link": "finger_1",
                "center_offset": [0, 0, 0],
                "human_hand_id": 8,
                "joint": ["joint_1"],
            },
        ],
        "joint": {"lower": [-1.0, -2.0], "upper": [1.0, 2.0]},
    }
    save_json(config, checkpoint / "config.json")
    state = IKModel([[0], [1]]).state_dict()
    for value in state.values():
        value.zero_()
    torch.save(
        {"state_dict": {f"ik_model.{key}": value for key, value in state.items()}},
        checkpoint / "last.ckpt",
    )
    return checkpoint


def test_exported_torchscript_matches_eager_inference(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)
    robot_spec = {
        "name": "two_finger",
        "hand_side": "right",
        "joint_order": ["joint_0", "joint_1"],
        "urdf_sha256": "a" * 64,
    }
    experiment = {
        "mocap": "webxr",
        "calibration": {
            "scale": 1.0,
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "outward_sign": 1,
        },
    }

    model_path, metadata_path = export_torchscript(
        checkpoint,
        robot_spec,
        tmp_path / "model.ts",
        experiment=experiment,
    )
    scripted = torch.jit.load(str(model_path))
    eager = GeoRTMethod(checkpoint, robot_spec, device="cpu")
    rng = np.random.default_rng(7)
    for timestamp in range(10):
        points = rng.normal(size=(21, 3)).astype(np.float32)
        actual = scripted(torch.from_numpy(points)).numpy()
        expected = eager.infer(
            CanonicalFrame(points, float(timestamp), "right")
        ).qpos
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    metadata = json.loads(metadata_path.read_text())
    assert metadata["joint_names"] == ["joint_0", "joint_1"]
    assert metadata["human_landmark_ids"] == [4, 8]
    assert metadata["hand_side"] == "right"
    assert metadata["mocap"] == "webxr"
    assert metadata["calibration"]["scale"] == 1.0
