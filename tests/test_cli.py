import json
import subprocess
import sys

import numpy as np
import pytest

from geort.artifacts import create_dataset, create_run
from geort.robots import load_robot
from geort.schema import CanonicalFrame, NamedCommand


def test_module_cli_exposes_the_four_user_workflows():
    result = subprocess.run(
        [sys.executable, "-m", "geort", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for command in ("collect", "train", "rollout", "evaluate"):
        assert command in result.stdout


def test_rollout_forwards_the_local_live_config(monkeypatch):
    import geort.rollout as rollout
    from geort.cli import main

    calls = {}
    monkeypatch.setattr(
        rollout,
        "run",
        lambda run, source, device, config: calls.update(
            run=run, source=source, device=device, config=config
        ),
    )

    main(
        [
            "rollout",
            "--run",
            "run-a",
            "--source",
            "live",
            "--config",
            ".geort/configs/manus_right.json",
        ]
    )

    assert calls == {
        "run": "run-a",
        "source": "live",
        "device": None,
        "config": ".geort/configs/manus_right.json",
    }


def test_live_rollout_requires_config_before_starting_viewer(monkeypatch):
    import geort.rollout as rollout
    from geort.cli import main

    monkeypatch.setattr(
        rollout,
        "run",
        lambda *args: pytest.fail("rollout started without capture config"),
    )

    with pytest.raises(ValueError, match="requires --config"):
        main(["rollout", "--run", "run-a", "--source", "live"])


def test_new_train_requires_dataset_and_robot():
    from geort.cli import main

    with pytest.raises(ValueError, match="requires --dataset and --robot"):
        main(["train"])


def _recording(valid=(True, True)):
    points = np.zeros((21, 3), dtype=np.float32)
    points[9] = [0, 0, 0.1]
    points[2] = [0, 0.03, 0]
    points[17] = [0, -0.03, 0]
    return [
        CanonicalFrame(points, float(index), "right", is_valid)
        for index, is_valid in enumerate(valid)
    ]


def _metadata():
    return {
        "mocap": "manus",
        "hand_side": "right",
        "calibration": {
            "scale": 1.0,
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "outward_sign": 1,
        },
    }


def test_run_metadata_rejects_absolute_robot_config_paths(tmp_path):
    from geort.cli import _run_config

    with pytest.raises(ValueError, match="project-relative"):
        _run_config(
            "dataset",
            str(tmp_path / "robot.json"),
            {"robot_fingerprint": "a" * 64, "hand_side": "right"},
            "geort",
            0,
        )


def test_collect_imports_a_recording_into_one_dataset(tmp_path, monkeypatch):
    from geort.cli import main

    monkeypatch.setenv("GEORT_HOME", str(tmp_path / ".geort"))
    config = tmp_path / "manus.json"
    config.write_text(json.dumps(_metadata()), encoding="utf-8")
    raw = np.stack([frame.points for frame in _recording()])
    input_path = tmp_path / "raw.npy"
    np.save(input_path, raw)

    dataset = main(
        [
            "collect",
            "--mocap",
            "manus",
            "--dataset",
            "manus_right_001",
            "--config",
            str(config),
            "--input",
            str(input_path),
        ]
    )

    assert dataset == tmp_path / ".geort" / "data" / "manus_right_001"
    assert {path.name for path in dataset.iterdir()} == {
        "raw.npz",
        "canonical.npy",
        "metadata.json",
    }
    assert json.loads((dataset / "metadata.json").read_text())["timestamps"] == [
        0.0,
        1 / 30,
    ]


def test_train_uses_only_valid_frames_and_writes_one_run(tmp_path, monkeypatch):
    import geort.trainer as trainer_module
    from geort.cli import main

    root = tmp_path / ".geort"
    monkeypatch.setenv("GEORT_HOME", str(root))
    frames = _recording((True, False, True))
    create_dataset(
        "manus_right_001",
        np.stack([frame.points for frame in frames]),
        frames,
        _metadata(),
    )
    calls = {}

    class Trainer:
        def __init__(self, robot, **kwargs):
            calls["robot"] = robot
            calls["constructor"] = kwargs

        def train(self, points, **kwargs):
            calls["points"] = points
            calls["train"] = kwargs
            checkpoint_dir = calls["constructor"]["checkpoint_dir"]
            (checkpoint_dir / "best.ckpt").write_bytes(b"checkpoint")
            return checkpoint_dir

    monkeypatch.setattr(trainer_module, "GeoRTTrainer", Trainer)

    run = main(
        [
            "train",
            "--dataset",
            "manus_right_001",
            "--robot",
            "wuji_right",
            "--run-id",
            "manus_wuji_geort_seed0",
            "--device",
            "cpu",
            "--epoch",
            "2",
            "--save-every",
            "0",
        ]
    )

    assert run == root / "runs" / "manus_wuji_geort_seed0"
    assert calls["points"].shape == (2, 21, 3)
    assert calls["constructor"]["checkpoint_dir"] == run / "checkpoints"
    assert calls["train"]["epoch"] == 2
    config = json.loads((run / "config.json").read_text())
    assert config["dataset"] == "manus_right_001"
    assert config["robot"] == "wuji_right"
    assert config["method"] == "geort"
    assert str(tmp_path) not in json.dumps(config)


def test_train_resumes_a_run_with_its_saved_settings(tmp_path, monkeypatch):
    import geort.trainer as trainer_module
    from geort.cli import main

    root = tmp_path / ".geort"
    monkeypatch.setenv("GEORT_HOME", str(root))
    frames = _recording()
    create_dataset(
        "manus_right_001",
        np.stack([frame.points for frame in frames]),
        frames,
        _metadata(),
    )
    robot = load_robot("wuji_right")
    training = {
        "seed": 7,
        "val_fraction": 0.2,
        "coverage_samples": 9,
        "coverage_batch_size": 3,
        "gesture_batch_size": 2,
        "save_every": 4,
        "direction_sigma": 0.005,
        "flatness_sigma": 0.002,
        "w_chamfer": 80.0,
        "w_curvature": 1.0,
        "w_pinch": 1000.0,
        "w_collision": 1e-4,
    }
    run = create_run(
        "manus_wuji_geort_seed7",
        {
            "dataset": "manus_right_001",
            "robot": "wuji_right",
            "robot_fingerprint": robot["robot_fingerprint"],
            "method": "geort",
            "seed": 7,
            "hand_side": "right",
            "method_options": {},
            "training": training,
        },
    )
    calls = {}

    class Trainer:
        def __init__(self, loaded_robot, **kwargs):
            calls["robot"] = loaded_robot
            calls["constructor"] = kwargs

        def train(self, points, **kwargs):
            calls["points"] = points
            calls["train"] = kwargs

    monkeypatch.setattr(trainer_module, "GeoRTTrainer", Trainer)

    resumed = main(
        [
            "train",
            "--resume",
            "manus_wuji_geort_seed7",
            "--epoch",
            "500",
            "--device",
            "cpu",
        ]
    )

    assert resumed == run
    assert calls["constructor"]["checkpoint_dir"] == run / "checkpoints"
    assert calls["points"].shape == (2, 21, 3)
    assert calls["train"] == {
        **training,
        "epoch": 500,
        "resume": run / "checkpoints",
    }


def test_train_rejects_a_dataset_without_two_valid_frames(tmp_path, monkeypatch):
    from geort.cli import main

    monkeypatch.setenv("GEORT_HOME", str(tmp_path / ".geort"))
    frames = _recording((False, False))
    create_dataset(
        "invalid_recording",
        np.stack([frame.points for frame in frames]),
        frames,
        _metadata(),
    )

    with pytest.raises(ValueError, match="at least two valid frames"):
        main(
            [
                "train",
                "--dataset",
                "invalid_recording",
                "--robot",
                "allegro_right",
            ]
        )


def test_evaluate_runs_dexpilot_inference_and_metrics_together(
    tmp_path, monkeypatch
):
    import geort.cli as cli

    root = tmp_path / ".geort"
    monkeypatch.setenv("GEORT_HOME", str(root))
    frames = _recording()
    create_dataset(
        "manus_right_001",
        np.stack([frame.points for frame in frames]),
        frames,
        _metadata(),
    )

    class Method:
        def infer(self, frame):
            return NamedCommand(
                tuple(load_robot("allegro_right")["joint_order"]),
                np.zeros(16, dtype=np.float32),
                frame.timestamp,
            )

    class Hand:
        def get_joint_limit(self):
            return -np.ones(16), np.ones(16)

        def initialize_keypoint(self, names, offsets):
            self.tip_count = len(names)

        def keypoint_from_qpos(self, qpos, ret_vec=False):
            return np.zeros((self.tip_count, 3))

        def is_self_collision(self, qpos):
            return False

    monkeypatch.setattr(cli, "make_method", lambda *args, **kwargs: Method())
    monkeypatch.setattr(cli, "_build_hand", lambda robot: Hand())

    metrics = cli.main(
        [
            "evaluate",
            "--dataset",
            "manus_right_001",
            "--robot",
            "allegro_right",
            "--method",
            "dexpilot",
            "--run-id",
            "manus_allegro_dexpilot_seed0",
        ]
    )

    outputs = root / "runs" / "manus_allegro_dexpilot_seed0" / "outputs"
    assert {"qpos.npz", "latency.npy", "metrics.json"} <= {
        path.name for path in outputs.iterdir()
    }
    assert metrics["joint_limit_violations"] == 0
    repeated = cli.main(
        ["evaluate", "--run", "manus_allegro_dexpilot_seed0"]
    )
    assert repeated["joint_limit_violations"] == 0
