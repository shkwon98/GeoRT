import torch

from geort.export import load_model
from geort.model import IKModel
from geort.utils.config_utils import save_json


def _write_checkpoint(checkpoint_root):
    checkpoint_dir = checkpoint_root / "two_finger"
    checkpoint_dir.mkdir(parents=True)
    save_json(
        {
            "joint_order": ["joint_0", "joint_1"],
            "fingertip_link": [
                {"link": "finger_0", "center_offset": [
                    0, 0, 0], "human_hand_id": 0, "joint": ["joint_0"]},
                {"link": "finger_1", "center_offset": [
                    0, 0, 0], "human_hand_id": 1, "joint": ["joint_1"]},
            ],
            "joint": {"lower": [-1.0, -2.0], "upper": [1.0, 2.0]},
        },
        checkpoint_dir / "config.json",
    )
    model = IKModel([[0], [1]])
    last_state = model.state_dict()
    for value in last_state.values():
        value.zero_()
    torch.save(
        {"state_dict": {f"ik_model.{key}": value for key, value in last_state.items()}},
        checkpoint_dir / "last.ckpt",
    )
    epoch_zero_state = model.state_dict()
    epoch_zero_state["nets.0.0.bias"].fill_(0.25)
    torch.save(
        {
            "state_dict": {
                f"ik_model.{key}": value for key, value in epoch_zero_state.items()
            }
        },
        checkpoint_dir / "epoch=0000.ckpt",
    )
    best_state = model.state_dict()
    best_state["nets.0.0.bias"].fill_(0.5)
    torch.save(
        {"state_dict": {f"ik_model.{key}": value for key, value in best_state.items()}},
        checkpoint_dir / "best.ckpt",
    )
    return checkpoint_dir


def test_load_model_is_cpu_safe_and_selects_exact_epoch(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = _write_checkpoint(checkpoint_root)
    monkeypatch.chdir(tmp_path)

    model = load_model(checkpoint_dir, device="cpu")
    last = load_model(checkpoint_dir, epoch=-1, device="cpu")
    epoch_zero = load_model("two_finger", epoch=0,
                            checkpoint_root=checkpoint_root, device="cpu")

    assert model.device.type == "cpu"
    assert torch.all(model.model.nets[0][0].bias == 0.5)
    assert torch.all(last.model.nets[0][0].bias == 0)
    assert torch.all(epoch_zero.model.nets[0][0].bias == 0.25)
