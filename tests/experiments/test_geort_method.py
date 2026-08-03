import numpy as np
import pytest

from experiments.methods.geort import GeoRTMethod, train
from experiments.schema import CanonicalFrame


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
            "epoch": 1,
        },
    )

    assert result == tmp_path / "checkpoint"
    assert calls["config"] is spec
    assert calls["train"]["coverage_samples"] == 64
    assert calls["train"]["coverage_batch_size"] == 8
    assert calls["train"]["gesture_batch_size"] == 4


def test_train_rejects_unknown_options(tmp_path):
    with pytest.raises(ValueError, match="unknown GeoRT options"):
        train(tmp_path / "canonical.npy", {}, tmp_path, {"epoh": 1})
