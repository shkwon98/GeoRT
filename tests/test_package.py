import importlib
import sys
import types

import numpy as np
import torch

from geort.utils.config_utils import get_config
from geort.utils.path import (
    get_bundled_fk_checkpoint,
    get_checkpoint_root,
    get_data_root,
    get_hand_landmarker_path,
    get_human_data,
    get_human_data_output_path,
    resolve_resource_path,
)


def test_packaged_modules_and_runtime_resources_are_available():
    for module in ("geort.config", "geort.env", "geort.utils", "geort.mocap.camera"):
        assert importlib.import_module(module)

    config = get_config("allegro_right")
    assert resolve_resource_path(config["urdf_path"]).is_file()
    assert get_hand_landmarker_path().is_file()
    assert get_human_data("human_alex").is_file()
    assert get_bundled_fk_checkpoint("allegro_right").is_file()


def test_geort_home_keeps_outputs_outside_package(monkeypatch, tmp_path):
    monkeypatch.setenv("GEORT_HOME", str(tmp_path))

    assert get_data_root() == tmp_path / "data"
    assert get_checkpoint_root() == tmp_path / "checkpoint"
    assert get_human_data_output_path("recording") == tmp_path / "data" / "recording"


def test_hand_builder_resolves_the_packaged_urdf(monkeypatch):
    fake_sapien = types.ModuleType("sapien")
    fake_core = types.ModuleType("sapien.core")
    fake_utils = types.ModuleType("sapien.utils")
    fake_utils.Viewer = object
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)
    monkeypatch.setitem(sys.modules, "sapien.core", fake_core)
    monkeypatch.setitem(sys.modules, "sapien.utils", fake_utils)
    monkeypatch.delitem(sys.modules, "geort.env.hand", raising=False)
    hand_module = importlib.import_module("geort.env.hand")
    captured = {}

    def fake_init(self, **kwargs):
        self.engine = None
        self.scene = None
        captured.update(kwargs)

    monkeypatch.setattr(hand_module.HandKinematicModel, "__init__", fake_init)
    hand_module.HandKinematicModel.build_from_config(get_config("allegro_right"))

    assert captured["hand_urdf"] == str(resolve_resource_path("assets/allegro_right/allegro_hand_right.urdf"))


def test_trainer_uses_bundled_fk_before_training_a_new_cache(tmp_path, monkeypatch):
    from geort.trainer import GeoRTTrainer

    class Hand:
        def get_joint_limit(self):
            return np.full(16, -1.0), np.full(16, 1.0)

    trainer = GeoRTTrainer.__new__(GeoRTTrainer)
    trainer.config = get_config("allegro_right")
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path
    trainer.hand = Hand()
    monkeypatch.setattr(
        trainer,
        "get_robot_kinematics_dataset",
        lambda: (_ for _ in ()).throw(AssertionError("bundled FK was not used")),
    )

    model = trainer.get_robot_neural_fk_model()

    assert model(torch.zeros(2, 16)).shape == (2, 4, 3)
    assert not any(parameter.requires_grad for parameter in model.parameters())
