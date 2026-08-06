import json

import numpy as np
import pytest

import experiments.run as run_module
from experiments.run import build_parser, load_experiment, main
from experiments.schema import NamedCommand


def test_cli_exposes_complete_workflow_and_rejects_side_mismatch(tmp_path):
    parser = build_parser()
    assert parser.parse_args(
        ["collect", "--config", "x.json", "--input", "raw.npy"]
    ).command == "collect"
    assert parser.parse_args(["train", "--run-dir", "run"]).command == "train"
    assert parser.parse_args(
        ["export", "--run-dir", "run"]).command == "export"
    assert parser.parse_args(["infer", "--run-dir", "run"]).command == "infer"
    assert parser.parse_args(
        ["evaluate", "--run-dir", "run"]).command == "evaluate"

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "mocap": "webxr",
                "method": "geort",
                "robot": "allegro_right",
                "hand_side": "left",
                "calibration": {
                    "scale": 1.0,
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "outward_sign": 1,
                },
                "method_options": {},
            }
        )
    )
    with pytest.raises(ValueError, match="hand_side"):
        load_experiment(path)


def test_collect_infer_and_evaluate_create_complete_run(tmp_path, monkeypatch):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "mocap": "webxr",
                "method": "dexpilot",
                "robot": "allegro_right",
                "hand_side": "right",
                "calibration": {
                    "scale": 1.0,
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "outward_sign": 1,
                },
                "method_options": {},
            }
        )
    )
    raw = np.zeros((2, 25, 3), dtype=np.float32)
    raw[:, 11] = [0, 0, 0.1]
    raw[:, 2] = [0, 0.03, 0]
    raw[:, 21] = [0, -0.03, 0]
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, raw)
    monkeypatch.setenv("GEORT_HOME", str(tmp_path / "results"))

    class Method:
        def __init__(self, robot):
            self.robot = robot

        def infer(self, frame):
            return NamedCommand(
                tuple(self.robot["joint_order"]),
                np.zeros(len(self.robot["joint_order"]), dtype=np.float32),
                frame.timestamp,
            )

    class Hand:
        def __init__(self, robot):
            self.robot = robot

        def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
            self.tip_count = len(keypoint_link_names)

        def get_joint_limit(self):
            size = len(self.robot["joint_order"])
            return -np.ones(size), np.ones(size)

        def keypoint_from_qpos(self, qpos, ret_vec=False):
            return np.zeros((self.tip_count, 3))

        def is_self_collision(self, qpos):
            return False

    monkeypatch.setattr(
        run_module,
        "_make_method",
        lambda config, robot, checkpoint=None: Method(robot),
    )
    monkeypatch.setattr(run_module, "_build_hand", Hand)

    main(
        [
            "collect",
            "--config",
            str(config_path),
            "--input",
            str(raw_path),
            "--run-id",
            "run-a",
        ]
    )
    run_dir = tmp_path / "results" / "runs" / "run-a"
    main(["infer", "--run-dir", str(run_dir)])
    main(["evaluate", "--run-dir", str(run_dir)])

    expected = {
        "config.json",
        "robot.json",
        "versions.json",
        "raw.npz",
        "raw_metadata.json",
        "canonical.npy",
        "canonical_metadata.json",
        "qpos.npz",
        "latency.npy",
        "metrics.json",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
