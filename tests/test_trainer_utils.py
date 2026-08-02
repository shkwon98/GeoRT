import importlib
import json
import random
import sys

import numpy as np
import pytest
import torch


def test_trainer_imports_without_sapien(monkeypatch):
    sys.modules.pop("geort.trainer", None)
    monkeypatch.setitem(sys.modules, "sapien", None)
    monkeypatch.setitem(sys.modules, "sapien.core", None)

    assert importlib.import_module("geort.trainer")


def test_pinch_correspondence_uses_full_batch_expectation():
    from geort.loss import pinch_correspondence_loss

    human_keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]]
    )
    robot_keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]]
    )

    assert torch.isclose(pinch_correspondence_loss(human_keypoints, robot_keypoints), torch.tensor(2.0))


def test_set_seed_repeats_python_numpy_and_torch_values():
    from geort.trainer import set_seed

    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_restore_rng_state_replays_next_values():
    from geort.trainer import capture_rng_state, restore_rng_state, set_seed

    set_seed(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.random()
    np.random.rand()
    torch.rand(1)
    restore_rng_state(state)

    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_restore_rng_state_moves_loaded_rng_tensors_to_cpu(monkeypatch):
    from geort.trainer import restore_rng_state

    class LoadedRngTensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return f"cpu:{self.value}"

    restored = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda state: restored.setdefault("torch", state))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.setdefault("cuda", states))

    restore_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": LoadedRngTensor("torch"),
            "cuda": [LoadedRngTensor("cuda:0"), LoadedRngTensor("cuda:1")],
        }
    )

    assert restored == {"torch": "cpu:torch", "cuda": ["cpu:cuda:0", "cpu:cuda:1"]}


def test_training_state_round_trip_restores_model_optimizer_and_next_epoch(tmp_path):
    from geort.trainer import load_training_state, save_training_state

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    loss = model(torch.ones(1, 1)).sum()
    loss.backward()
    optimizer.step()
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    expected_step = next(iter(optimizer.state.values()))["momentum_buffer"].clone()
    checkpoint_path = tmp_path / "state.pt"

    save_training_state(checkpoint_path, model, optimizer, epoch=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    optimizer.param_groups[0]["lr"] = 0.1
    optimizer.state.clear()

    assert load_training_state(checkpoint_path, model, optimizer, "cpu") == 4
    assert all(torch.equal(parameter, expected) for parameter, expected in zip(model.parameters(), expected_parameters))
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert torch.equal(next(iter(optimizer.state.values()))["momentum_buffer"], expected_step)


def test_append_metrics_writes_jsonl_with_scalar_tensors(tmp_path):
    from geort.trainer import append_metrics

    path = tmp_path / "metrics.jsonl"
    append_metrics(path, {"epoch": 2, "loss": torch.tensor(1.25), "lr": 0.01})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"epoch": 2, "loss": 1.25, "lr": 0.01}
    ]


def test_split_aligned_frames_is_deterministic_and_keeps_whole_frames():
    from geort.trainer import split_aligned_frames

    frames = np.arange(10 * 3 * 3, dtype=np.float32).reshape(10, 3, 3)
    train, validation = split_aligned_frames(frames, val_fraction=0.3, seed=19)
    repeated_train, repeated_validation = split_aligned_frames(frames, val_fraction=0.3, seed=19)

    np.testing.assert_array_equal(train, repeated_train)
    np.testing.assert_array_equal(validation, repeated_validation)
    assert len(train) == 7
    assert len(validation) == 3
    original_frames = {frame.tobytes() for frame in frames}
    train_frames = {frame.tobytes() for frame in train}
    validation_frames = {frame.tobytes() for frame in validation}
    assert train_frames | validation_frames == original_frames
    assert train_frames.isdisjoint(validation_frames)


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_split_aligned_frames_rejects_invalid_validation_fraction(fraction):
    from geort.trainer import split_aligned_frames

    with pytest.raises(ValueError, match="val_fraction"):
        split_aligned_frames(np.zeros((3, 2, 3)), fraction, seed=0)


def test_split_aligned_frames_requires_two_frames_for_validation():
    from geort.trainer import split_aligned_frames

    with pytest.raises(ValueError, match="two frames"):
        split_aligned_frames(np.zeros((1, 2, 3)), 0.5, seed=0)


def test_paper_training_defaults_match_recommended_weights():
    from geort.trainer import PAPER_DEFAULTS

    assert PAPER_DEFAULTS == {
        "w_chamfer": 80.0,
        "w_curvature": 1.0,
        "w_pinch": 1000.0,
        "w_collision": 1e-4,
    }


def test_training_cli_exposes_reproducibility_and_path_options():
    from geort.trainer import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--device", "cpu",
            "--data-dir", "/tmp/data",
            "--checkpoint-dir", "/tmp/checkpoints",
            "--seed", "7",
            "--val-fraction", "0.25",
            "--epoch", "12",
            "--resume", "experiment",
        ]
    )

    assert args.device == "cpu"
    assert args.data_dir == "/tmp/data"
    assert args.checkpoint_dir == "/tmp/checkpoints"
    assert args.seed == 7
    assert args.val_fraction == 0.25
    assert args.epoch == 12
    assert args.resume == "experiment"


def test_resume_config_validation_accepts_match_and_names_mismatch():
    from geort.trainer import validate_resume_config

    current = {
        "name": "hand",
        "urdf_path": "hand.urdf",
        "base_link": "base",
        "joint_order": ["joint"],
        "fingertip_link": [{"link": "tip", "joint": ["joint"], "human_hand_id": 1}],
        "joint": {"lower": [0.0], "upper": [1.0]},
    }
    metadata = {
        "human_data": "/data/human.npy",
        "seed": 7,
        "val_fraction": 0.2,
        **PAPER_DEFAULTS_FOR_TEST,
        "direction_sigma": 0.005,
        "flatness_sigma": 0.002,
    }
    saved = {**current, "training": metadata}

    validate_resume_config(current, saved, metadata)
    mismatched = {**metadata, "seed": 8}
    with pytest.raises(ValueError, match="seed"):
        validate_resume_config(current, saved, mismatched)


def test_resume_validation_happens_before_stream_construction(tmp_path, monkeypatch):
    from geort.trainer import GeoRTTrainer

    config = _trainer_test_config()
    trainer = GeoRTTrainer.__new__(GeoRTTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path
    trainer.hand = _TrainerTestHand()
    human_path = tmp_path / "human.npy"
    np.save(human_path, np.zeros((2, 1, 3), dtype=np.float32))
    resume_dir = tmp_path / "experiment"
    resume_dir.mkdir()
    saved = _saved_trainer_config(config, human_path, val_fraction=0.0)
    saved["name"] = "different-hand"
    (resume_dir / "config.json").write_text(json.dumps(saved))
    monkeypatch.setattr(trainer, "_build_streams", lambda *args, **kwargs: pytest.fail("streams built"))

    with pytest.raises(ValueError, match="name"):
        trainer.train(human_path, epoch=1, val_fraction=0.0, resume=resume_dir)


def test_resume_does_not_overwrite_existing_config(tmp_path, monkeypatch):
    from geort.model import CollisionClassifier, FKModel, IKModel
    from geort.trainer import GeoRTTrainer, save_training_state

    config = _trainer_test_config()
    trainer = GeoRTTrainer.__new__(GeoRTTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path
    trainer.hand = _TrainerTestHand()
    human_path = tmp_path / "human.npy"
    np.save(human_path, np.zeros((2, 1, 3), dtype=np.float32))
    resume_dir = tmp_path / "experiment"
    resume_dir.mkdir()
    saved = _saved_trainer_config(config, human_path, val_fraction=0.0)
    config_path = resume_dir / "config.json"
    original_config_text = json.dumps(saved)
    config_path.write_text(original_config_text)
    model = IKModel([[0, 1]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_training_state(resume_dir / "training_state.pth", model, optimizer, epoch=0)
    torch.save(FKModel([[0, 1]]).state_dict(), resume_dir / "fk_model.pth")
    torch.save(CollisionClassifier(2).state_dict(), resume_dir / "collision_model.pth")

    monkeypatch.setattr(trainer, "_build_streams", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(trainer, "get_robot_neural_fk_model", lambda: pytest.fail("used mutable FK cache"))
    monkeypatch.setattr(trainer, "get_collision_classifier", lambda: pytest.fail("used mutable collision cache"))
    monkeypatch.setattr(trainer, "get_robot_pointcloud", lambda names: np.zeros((1, 2, 3)))

    assert trainer.train(human_path, epoch=1, val_fraction=0.0, resume=resume_dir) == resume_dir
    assert config_path.read_text() == original_config_text


def test_resume_requires_frozen_model_snapshots_before_building_streams(tmp_path, monkeypatch):
    from geort.model import IKModel
    from geort.trainer import GeoRTTrainer, save_training_state

    config = _trainer_test_config()
    trainer = GeoRTTrainer.__new__(GeoRTTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path
    trainer.hand = _TrainerTestHand()
    human_path = tmp_path / "human.npy"
    np.save(human_path, np.zeros((2, 1, 3), dtype=np.float32))
    resume_dir = tmp_path / "experiment"
    resume_dir.mkdir()
    (resume_dir / "config.json").write_text(json.dumps(_saved_trainer_config(config, human_path, 0.0)))
    model = IKModel([[0, 1]])
    save_training_state(
        resume_dir / "training_state.pth",
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        epoch=0,
    )
    monkeypatch.setattr(trainer, "_build_streams", lambda *args, **kwargs: pytest.fail("streams built"))

    with pytest.raises(FileNotFoundError, match="fk_model.pth"):
        trainer.train(human_path, epoch=1, val_fraction=0.0, resume=resume_dir)


def test_resolve_human_data_supports_nested_filename_in_data_dir(tmp_path):
    from geort.trainer import _resolve_human_data

    path = tmp_path / "sub" / "sample.npy"
    path.parent.mkdir()
    np.save(path, np.zeros((1, 1, 3)))

    assert _resolve_human_data("sub/sample.npy", tmp_path) == path


def test_build_streams_routes_coverage_and_aligned_gestures(monkeypatch):
    from geort.dataset import MultiPointDataset
    from geort.trainer import GeoRTTrainer

    frames = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    captured = {}

    def fake_from_points(points, n):
        captured["points"] = points.copy()
        return MultiPointDataset(points)

    monkeypatch.setattr(MultiPointDataset, "from_points", staticmethod(fake_from_points))
    coverage_loader, gesture_loader = GeoRTTrainer._build_streams(frames, shuffle=False)

    np.testing.assert_array_equal(captured["points"], frames.transpose(1, 0, 2))
    np.testing.assert_array_equal(coverage_loader.dataset.points, frames.transpose(1, 0, 2))
    np.testing.assert_array_equal(gesture_loader.dataset.frames, frames)


def test_build_streams_avoids_singleton_gesture_batch(monkeypatch):
    from geort.dataset import MultiPointDataset
    from geort.trainer import GeoRTTrainer

    frames = np.zeros((2049, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        MultiPointDataset,
        "from_points",
        staticmethod(lambda points, n: MultiPointDataset(points)),
    )

    _, gesture_loader = GeoRTTrainer._build_streams(frames, shuffle=False)

    assert gesture_loader.batch_size == 2047
    assert [len(batch) for batch in gesture_loader] == [2047, 2]


def test_train_rejects_single_aligned_frame_before_stream_build(tmp_path, monkeypatch):
    from geort.trainer import GeoRTTrainer

    trainer = GeoRTTrainer.__new__(GeoRTTrainer)
    trainer.config = _trainer_test_config()
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path
    trainer.hand = _TrainerTestHand()
    human_path = tmp_path / "human.npy"
    np.save(human_path, np.zeros((1, 1, 3), dtype=np.float32))
    monkeypatch.setattr(trainer, "_build_streams", lambda *args, **kwargs: pytest.fail("streams built"))

    with pytest.raises(ValueError, match="at least two"):
        trainer.train(human_path, epoch=1, val_fraction=0.0)


def test_fresh_training_rng_is_independent_of_auxiliary_cache_work(tmp_path, monkeypatch):
    import geort.trainer as trainer_module

    human_path = tmp_path / "human.npy"
    np.save(human_path, np.zeros((2, 1, 3), dtype=np.float32))
    initial_states = []
    first_training_values = []

    def capturing_ik(**kwargs):
        model = torch.nn.Linear(2, 2)
        initial_states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(trainer_module, "IKModel", capturing_ik)

    for run, consume_rng in enumerate((True, False)):
        trainer = trainer_module.GeoRTTrainer.__new__(trainer_module.GeoRTTrainer)
        trainer.config = _trainer_test_config()
        trainer.device = torch.device("cpu")
        trainer.checkpoint_dir = tmp_path / f"checkpoints-{run}"
        trainer.hand = _TrainerTestHand()
        monkeypatch.setattr(trainer, "_build_streams", lambda *args, **kwargs: ([], []))

        def auxiliary_model():
            if consume_rng:
                random.random()
                np.random.rand()
                torch.rand(3)
            return torch.nn.Linear(1, 1)

        def robot_points(names):
            if consume_rng:
                random.random()
                np.random.rand()
                torch.rand(3)
            return np.zeros((1, 2, 3))

        def capture_training_rng(*args, **kwargs):
            first_training_values.append((random.random(), np.random.rand(), torch.rand(1).item()))
            return {name: 0.0 for name in ("total", "direction", "chamfer", "curvature", "pinch", "collision")}

        monkeypatch.setattr(trainer, "get_robot_neural_fk_model", auxiliary_model)
        monkeypatch.setattr(trainer, "get_collision_classifier", lambda: torch.nn.Linear(1, 1))
        monkeypatch.setattr(trainer, "get_robot_pointcloud", robot_points)
        monkeypatch.setattr(trainer, "_run_epoch", capture_training_rng)

        trainer.train(human_path, epoch=1, seed=23, val_fraction=0.0)

    assert all(
        torch.equal(initial_states[0][name], initial_states[1][name])
        for name in initial_states[0]
    )
    assert first_training_values[0] == first_training_values[1]


def test_frozen_fk_and_classifier_keep_input_gradients():
    from geort.model import CollisionClassifier, FKModel
    from geort.trainer import _freeze_model

    fk = _freeze_model(FKModel([[0, 1]]))
    classifier = _freeze_model(CollisionClassifier(2))
    joint = torch.zeros(2, 2, requires_grad=True)
    (fk(joint).sum() + classifier(joint).sum()).backward()

    assert joint.grad is not None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in fk.parameters())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in classifier.parameters())


PAPER_DEFAULTS_FOR_TEST = {
    "w_chamfer": 80.0,
    "w_curvature": 1.0,
    "w_pinch": 1000.0,
    "w_collision": 1e-4,
}


def _trainer_test_config():
    return {
        "name": "hand",
        "urdf_path": "hand.urdf",
        "base_link": "base",
        "joint_order": ["joint0", "joint1"],
        "fingertip_link": [
            {
                "link": "tip",
                "joint": ["joint0", "joint1"],
                "center_offset": [0.0, 0.0, 0.0],
                "human_hand_id": 0,
            }
        ],
    }


class _TrainerTestHand:
    def get_joint_limit(self):
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])


def _saved_trainer_config(config, human_path, val_fraction):
    return {
        **config,
        "joint": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "training": {
            "human_data": str(human_path.resolve()),
            "seed": 0,
            "val_fraction": val_fraction,
            **PAPER_DEFAULTS_FOR_TEST,
            "direction_sigma": 0.005,
            "flatness_sigma": 0.002,
        },
    }
