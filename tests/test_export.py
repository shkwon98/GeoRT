import numpy as np
import pytest
import torch

from geort.export import load_model
from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.config_utils import save_json


def _write_checkpoint(checkpoint_root):
    checkpoint_dir = checkpoint_root / "two_finger"
    checkpoint_dir.mkdir(parents=True)
    save_json(
        {
            "joint_order": ["joint_0", "joint_1"],
            "fingertip_link": [
                {
                    "link": "finger_0",
                    "center_offset": [0, 0, 0],
                    "human_hand_id": 0,
                    "joint": ["joint_0"],
                },
                {
                    "link": "finger_1",
                    "center_offset": [0, 0, 0],
                    "human_hand_id": 1,
                    "joint": ["joint_1"],
                },
            ],
            "joint": {"lower": [-1.0, -2.0], "upper": [1.0, 2.0]},
        },
        checkpoint_dir / "config.json",
    )
    model = IKModel([[0], [1]])
    last_state = model.state_dict()
    for value in last_state.values():
        value.zero_()
    torch.save(last_state, checkpoint_dir / "last.pth")

    epoch_zero_state = model.state_dict()
    epoch_zero_state["nets.0.0.bias"].fill_(0.25)
    torch.save(epoch_zero_state, checkpoint_dir / "epoch_0.pth")
    return checkpoint_dir


def test_load_model_runs_on_cpu_from_non_repository_cwd(tmp_path, monkeypatch):
    checkpoint_dir = _write_checkpoint(tmp_path / "checkpoints")
    monkeypatch.chdir(tmp_path)

    model = load_model(checkpoint_dir, device="cpu")
    qpos = model.forward(np.zeros((2, 3), dtype=np.float32))

    assert model.device.type == "cpu"
    assert qpos.shape == (2,)
    np.testing.assert_allclose(qpos, [0.0, 0.0])


def test_load_model_selects_epoch_zero_and_last_by_exact_tag(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_root)

    epoch_zero = load_model("two_finger", epoch=0, checkpoint_root=checkpoint_root, device="cpu")
    last = load_model("two_finger", epoch=None, checkpoint_root=checkpoint_root, device="cpu")

    assert torch.all(epoch_zero.model.nets[0][0].bias == 0.25)
    assert torch.all(last.model.nets[0][0].bias == 0)
    with pytest.raises(FileNotFoundError, match="two"):
        load_model("two", checkpoint_root=checkpoint_root, device="cpu")


def test_load_model_names_missing_checkpoint_files(tmp_path):
    checkpoint_dir = tmp_path / "incomplete"
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="config.json"):
        load_model(checkpoint_dir, device="cpu")


def test_normalize_torch_preserves_input_device_and_dtype():
    formatter = HandFormatter([-2.0, 0.0], [2.0, 4.0])
    values = torch.tensor([0.0, 2.0], dtype=torch.float64)

    normalized = formatter.normalize_torch(values)

    assert normalized.device == values.device
    assert normalized.dtype == values.dtype
    assert torch.equal(normalized, torch.zeros_like(values))
