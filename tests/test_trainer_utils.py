import importlib
import json
import random
import sys

import numpy as np
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
