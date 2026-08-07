import numpy as np

from geort.metrics import evaluate_trajectory
from geort.schema import CanonicalFrame


class _Hand:
    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        self.names = keypoint_link_names

    def get_joint_limit(self):
        return np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        return np.array([[qpos[0], 0, 0], [qpos[1], 0, 0]])

    def is_self_collision(self, qpos):
        return bool(qpos[0] > 0.5)


def test_evaluate_trajectory_reports_common_metrics():
    points = np.zeros((21, 3), dtype=np.float32)
    points[4, 0] = 0.2
    points[8, 0] = 0.4
    frames = [
        CanonicalFrame(points, 0.0, "right"),
        CanonicalFrame(points, 0.1, "right"),
    ]
    robot = {
        "joint_order": ["a", "b"],
        "fingertip_link": [
            {"link": "thumb", "center_offset": [0, 0, 0], "human_hand_id": 4},
            {"link": "index", "center_offset": [0, 0, 0], "human_hand_id": 8},
        ],
    }

    metrics = evaluate_trajectory(
        frames,
        np.array([[0.2, 0.4], [0.6, 1.2]]),
        ("a", "b"),
        np.array([0.001, 0.003]),
        robot,
        _Hand(),
    )

    assert metrics["latency_ms_p50"] == 2.0
    assert metrics["latency_ms_p95"] == 2.9
    assert metrics["joint_limit_violations"] == 1
    assert metrics["self_collision_rate"] == 0.5
