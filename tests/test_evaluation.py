import numpy as np
import pytest

from geort.mocap import evaluation
from geort.mocap.evaluation import parse_args


def test_evaluation_parser_accepts_replay_data():
    args = parse_args([
        "--mocap", "replay",
        "--hand", "allegro_right",
        "--ckpt-tag", "/tmp/checkpoint",
        "--data", "human",
    ])

    assert args.mocap == "replay"
    assert args.data == "human"


def test_evaluation_parser_requires_replay_data():
    with pytest.raises(SystemExit):
        parse_args([
            "--mocap", "replay",
            "--hand", "allegro_right",
            "--ckpt-tag", "/tmp/checkpoint",
        ])


def test_apply_mocap_frame_updates_robot_and_overlay_together():
    points = np.arange(63, dtype=np.float32).reshape(21, 3)
    qpos = np.array([0.1, 0.2], dtype=np.float32)
    robot_points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    class Model:
        def forward(self, value):
            assert value is points
            return qpos

    class Hand:
        def set_qpos_target(self, value):
            self.target = value

        def keypoint_from_qpos(self, value, ret_vec=False):
            assert value is qpos and ret_vec
            return robot_points

    class Viewer:
        def set_mocap_overlay(self, *values):
            self.overlay = values

    hand = Hand()
    viewer = Viewer()

    evaluation._apply_mocap_frame(
        {"status": "recording", "result": points},
        Model(),
        hand,
        viewer,
        [4, 8],
    )

    assert hand.target is qpos
    assert viewer.overlay == (points, [4, 8], robot_points)
