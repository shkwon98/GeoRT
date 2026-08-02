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
