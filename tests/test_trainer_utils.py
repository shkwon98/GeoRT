import importlib
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset


pytestmark = [
    pytest.mark.filterwarnings("ignore:GPU available but not used.*"),
    pytest.mark.filterwarnings("ignore:The '.*_dataloader' does not have many workers.*"),
    pytest.mark.filterwarnings("ignore:Checkpoint directory .* exists and is not empty.*"),
    pytest.mark.filterwarnings("ignore:Found .* module\\(s\\) in eval mode.*"),
    pytest.mark.filterwarnings(
        "ignore:You defined a `validation_step` but have no `val_dataloader`.*"
    ),
]


def test_trainer_imports_without_sapien(monkeypatch):
    sys.modules.pop("geort.trainer", None)
    monkeypatch.setitem(sys.modules, "sapien", None)
    monkeypatch.setitem(sys.modules, "sapien.core", None)

    assert importlib.import_module("geort.trainer")


def test_paper_defaults_and_frame_split_are_deterministic():
    from geort.trainer import PAPER_DEFAULTS, split_aligned_frames

    frames = np.arange(10 * 3 * 3, dtype=np.float32).reshape(10, 3, 3)
    train, validation = split_aligned_frames(frames, val_fraction=0.3, seed=19)
    repeated_train, repeated_validation = split_aligned_frames(
        frames, val_fraction=0.3, seed=19)

    assert PAPER_DEFAULTS == {
        "w_chamfer": 80.0, "w_curvature": 1.0, "w_pinch": 1000.0, "w_collision": 1e-4}
    np.testing.assert_array_equal(train, repeated_train)
    np.testing.assert_array_equal(validation, repeated_validation)
    assert len(train) == 7
    assert len(validation) == 3


def test_lightning_checkpoint_restores_rng_state():
    from geort.model import CollisionClassifier, FKModel, IKModel
    from geort.trainer import GeoRTLightningModule, set_seed

    groups = [list(range(start, start + 4)) for start in range(0, 16, 4)]
    module = GeoRTLightningModule(
        IKModel(groups, hidden_dim=8),
        FKModel(groups, hidden_dim=8),
        CollisionClassifier(16, hidden_dim=8),
        torch.randn(4, 8, 3),
        {"w_chamfer": 80.0, "w_curvature": 1.0,
         "w_pinch": 1000.0, "w_collision": 1e-4},
        direction_sigma=0.005,
        flatness_sigma=0.002,
        validation_seed=0,
    )
    set_seed(17)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    next_values = (random.random(), np.random.rand(), torch.rand(1).item())

    module.on_load_checkpoint(checkpoint)

    assert (random.random(), np.random.rand(),
            torch.rand(1).item()) == next_values


def test_validation_rng_is_repeatable_without_advancing_training_rng():
    from geort import trainer

    trainer.set_seed(11)
    torch.rand(1)
    expected_next = torch.rand(1)

    trainer.set_seed(11)
    torch.rand(1)
    with trainer.fixed_rng(7):
        first_validation = torch.rand(3)
    actual_next = torch.rand(1)
    with trainer.fixed_rng(7):
        repeated_validation = torch.rand(3)

    assert torch.equal(first_validation, repeated_validation)
    assert torch.equal(actual_next, expected_next)


def test_trainer_separates_robot_cache_from_run_checkpoints(tmp_path, monkeypatch):
    from geort.env.hand import HandKinematicModel
    from geort.trainer import GeoRTTrainer

    monkeypatch.setenv("GEORT_HOME", str(tmp_path / ".geort"))
    monkeypatch.setattr(
        HandKinematicModel,
        "build_from_config",
        lambda config: object(),
    )
    robot = {"name": "wuji_right", "urdf_sha256": "a" * 64}
    checkpoint_dir = tmp_path / "run" / "checkpoints"

    trainer = GeoRTTrainer(robot, device="cpu", checkpoint_dir=checkpoint_dir)

    cache_dir = tmp_path / ".geort" / "cache" / f"wuji_right-{'a' * 12}"
    assert trainer.data_dir == cache_dir
    assert trainer.checkpoint_dir == checkpoint_dir
    assert trainer.get_fk_checkpoint_path() == (
        cache_dir / "fk_model_wuji_right.pth"
    )
    get_collision_checkpoint_path = getattr(
        trainer, "get_collision_checkpoint_path", lambda: None
    )
    assert get_collision_checkpoint_path() == (
        cache_dir / "collision_model_wuji_right.pth"
    )


def test_training_size_options_control_streams(monkeypatch):
    from geort.trainer import GeoRTTrainer, build_arg_parser

    args = build_arg_parser().parse_args([
        "--coverage-samples", "7",
        "--coverage-batch-size", "3",
        "--gesture-batch-size", "2",
        "--save-every", "7",
    ])
    frames = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

    monkeypatch.setattr(
        "geort.trainer.MultiPointDataset.from_points",
        lambda points, n: torch.utils.data.TensorDataset(
            torch.zeros(n, points.shape[0], 3)
        ),
    )
    coverage, gestures = GeoRTTrainer._build_streams(
        frames,
        shuffle=False,
        coverage_samples=args.coverage_samples,
        coverage_batch_size=args.coverage_batch_size,
        gesture_batch_size=args.gesture_batch_size,
    )

    assert len(coverage.dataset) == 7
    assert coverage.batch_size == 3
    assert gestures.batch_size == 2
    assert args.save_every == 7


class _CheckpointModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("validation/total", 4.0 - self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_lightning_checkpoint_policy_keeps_best_last_and_periodic(tmp_path):
    from geort.trainer import build_checkpoint_callbacks

    loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
    trainer = Trainer(
        accelerator="cpu",
        callbacks=build_checkpoint_callbacks(tmp_path, save_every=2),
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_epochs=5,
    )

    trainer.fit(_CheckpointModule(), loader, loader)

    assert {path.name for path in tmp_path.glob("*.ckpt")} == {
        "best.ckpt",
        "last.ckpt",
        "epoch=0001.ckpt",
        "epoch=0003.ckpt",
    }
    assert torch.load(
        tmp_path / "best.ckpt", map_location="cpu", weights_only=True
    )["epoch"] == 4


def test_save_every_zero_disables_periodic_checkpoints(tmp_path):
    from geort.trainer import build_checkpoint_callbacks

    callbacks = build_checkpoint_callbacks(tmp_path, save_every=0)

    assert len(callbacks) == 1


def test_checkpoint_policy_without_validation_only_keeps_last(tmp_path):
    from geort.trainer import build_checkpoint_callbacks

    loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
    trainer = Trainer(
        accelerator="cpu",
        callbacks=build_checkpoint_callbacks(
            tmp_path, save_every=0, has_validation=False
        ),
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_epochs=2,
    )

    trainer.fit(_CheckpointModule(), loader)

    assert {path.name for path in tmp_path.glob("*.ckpt")} == {"last.ckpt"}


def test_save_every_rejects_negative_values(tmp_path):
    from geort.trainer import build_checkpoint_callbacks

    with pytest.raises(ValueError, match="non-negative"):
        build_checkpoint_callbacks(tmp_path, save_every=-1)


def test_geort_trainer_writes_and_resumes_lightning_checkpoints(tmp_path):
    from geort.model import CollisionClassifier, FKModel
    from geort.trainer import GeoRTTrainer

    finger_count = 4
    joint_count = finger_count * 4
    groups = [list(range(start, start + 4))
              for start in range(0, joint_count, 4)]
    human_ids = [4, 8, 12, 16]
    trainer = GeoRTTrainer.__new__(GeoRTTrainer)
    trainer.config = {
        "name": "synthetic_right",
        "urdf_path": "synthetic.urdf",
        "base_link": "base",
        "joint_order": [f"joint_{index}" for index in range(joint_count)],
        "fingertip_link": [
            {
                "link": f"tip_{index}",
                "center_offset": [0, 0, 0],
                "human_hand_id": human_id,
                "joint": [f"joint_{joint}" for joint in group],
            }
            for index, (human_id, group) in enumerate(zip(human_ids, groups))
        ],
    }
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.hand = SimpleNamespace(
        get_joint_limit=lambda: (
            -np.ones(joint_count, dtype=np.float32),
            np.ones(joint_count, dtype=np.float32),
        )
    )
    trainer.get_robot_neural_fk_model = lambda: FKModel(
        groups, hidden_dim=8
    ).eval()
    trainer.get_collision_classifier = lambda: CollisionClassifier(
        joint_count, hidden_dim=8
    ).eval()
    trainer.get_robot_pointcloud = lambda links: np.random.default_rng(3).normal(
        size=(finger_count, 8, 3)
    ).astype(np.float32)

    human_data = tmp_path / "human.npy"
    np.save(
        human_data,
        np.random.default_rng(5).normal(size=(8, 21, 3)).astype(np.float32),
    )
    options = {
        "epoch": 2,
        "save_every": 2,
        "coverage_samples": 4,
        "coverage_batch_size": 4,
        "gesture_batch_size": 4,
    }

    save_dir = trainer.train(human_data, **options)

    assert {path.name for path in save_dir.glob("*.ckpt")} == {
        "best.ckpt", "last.ckpt", "epoch=0001.ckpt"
    }
    assert not list(save_dir.glob("*.pth"))
    assert json.loads((save_dir / "config.json").read_text())[
        "training"
    ]["save_every"] == 2

    trainer.train(human_data, resume=save_dir, **{**options, "epoch": 3})

    assert torch.load(
        save_dir / "last.ckpt", map_location="cpu", weights_only=True
    )["epoch"] == 2
    assert len(list((save_dir / "logs").glob("version_*/metrics.csv"))) == 2


@pytest.mark.parametrize("value", [0, -1])
def test_training_size_options_reject_non_positive_values(value):
    from geort.trainer import GeoRTTrainer

    frames = np.zeros((4, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        GeoRTTrainer._build_streams(
            frames,
            shuffle=False,
            coverage_samples=value,
        )


@pytest.mark.parametrize("finger_count", [4, 5])
def test_one_synthetic_lightning_epoch_supports_robot_finger_counts(finger_count):
    from geort.model import CollisionClassifier, FKModel, IKModel
    from geort.trainer import GeoRTLightningModule

    groups = [
        list(range(start, start + 4))
        for start in range(0, finger_count * 4, 4)
    ]
    module = GeoRTLightningModule(
        IKModel(groups, hidden_dim=8),
        FKModel(groups, hidden_dim=8),
        CollisionClassifier(finger_count * 4, hidden_dim=8),
        torch.randn(finger_count, 8, 3),
        {
            "w_chamfer": 80.0,
            "w_curvature": 1.0,
            "w_pinch": 1000.0,
            "w_collision": 1e-4,
        },
        direction_sigma=0.005,
        flatness_sigma=0.002,
        validation_seed=0,
    )
    coverage = DataLoader(torch.randn(4, finger_count, 3), batch_size=4)
    gestures = DataLoader(torch.randn(4, finger_count, 3), batch_size=4)
    trainer = Trainer(
        accelerator="cpu",
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_epochs=1,
    )

    train_loader = CombinedLoader(
        {"coverage": coverage, "gesture": gestures}, mode="max_size_cycle"
    )
    validation_loader = CombinedLoader(
        {"coverage": coverage, "gesture": gestures}, mode="max_size_cycle"
    )
    trainer.fit(module, train_loader, validation_loader)

    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["validation/total"])
