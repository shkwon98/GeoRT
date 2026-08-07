import json

import numpy as np

from geort.rollout import _replay_interval, apply_frame, collect_live
from geort.schema import CanonicalFrame, NamedCommand


def test_apply_frame_updates_robot_and_mocap_view_together():
    points = np.arange(63, dtype=np.float32).reshape(21, 3)
    qpos = np.array([0.1, 0.2], dtype=np.float32)
    robot_points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    class Method:
        def infer(self, frame):
            assert frame.points is points
            return NamedCommand(("a", "b"), qpos, frame.timestamp)

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
    apply_frame(
        CanonicalFrame(points, 1.0, "right"),
        Method(),
        hand,
        viewer,
        [4, 8],
    )

    assert hand.target is qpos
    assert viewer.overlay == (points, [4, 8], robot_points)


def test_collect_live_keeps_raw_observations_and_closes_source(
    tmp_path, monkeypatch
):
    import geort.rollout as rollout

    capture_config = tmp_path / "manus.json"
    capture_config.write_text(json.dumps({"topic": "/manus_quats"}))
    observations = [np.zeros((21, 3)), np.ones((21, 3))]

    class Source:
        def __init__(self):
            self.results = [
                {"status": "recording", "result": observations[0], "timestamp": 1.0},
                {"status": "recording", "result": observations[1], "timestamp": 1.1},
                {"status": "quit", "result": None},
            ]

        def get(self):
            return self.results.pop(0)

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(rollout, "_live_source", lambda *args: source)

    raw, timestamps = collect_live("manus", capture_config)

    np.testing.assert_array_equal(raw, observations)
    np.testing.assert_array_equal(timestamps, [1.0, 1.1])
    assert source.closed


def test_replay_preserves_recorded_frame_interval():
    points = np.zeros((21, 3), dtype=np.float32)
    frames = [
        CanonicalFrame(points, 0.0, "right"),
        CanonicalFrame(points, 1.0, "right"),
    ]

    assert _replay_interval(frames, 0) == 1.0
    assert _replay_interval(frames, 1) == 1 / 30
