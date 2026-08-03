import importlib
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


def test_training_state_restores_model_and_rng(tmp_path):
    from geort.trainer import load_training_state, save_training_state, set_seed

    set_seed(17)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    expected = [parameter.detach().clone() for parameter in model.parameters()]
    path = tmp_path / "state.pth"
    save_training_state(path, model, optimizer, epoch=3)
    next_values = (random.random(), np.random.rand(), torch.rand(1).item())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)

    assert load_training_state(path, model, optimizer, "cpu") == 4
    assert all(torch.equal(parameter, value)
               for parameter, value in zip(model.parameters(), expected))
    assert (random.random(), np.random.rand(),
            torch.rand(1).item()) == next_values


def test_training_size_options_control_streams(monkeypatch):
    from geort.trainer import GeoRTTrainer, build_arg_parser

    args = build_arg_parser().parse_args([
        "--coverage-samples", "7",
        "--coverage-batch-size", "3",
        "--gesture-batch-size", "2",
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
