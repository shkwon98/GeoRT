from types import SimpleNamespace

import numpy as np

from geort.methods import DexPilotMethod, GeoRTMethod
from geort.schema import CanonicalFrame


class _Model:
    def forward(self, points):
        assert points.shape == (21, 3)
        return np.array([0.2, -0.1], dtype=np.float32)


def test_geort_returns_a_robot_named_command():
    method = GeoRTMethod.__new__(GeoRTMethod)
    method.model = _Model()
    method.joint_names = ("joint_b", "joint_a")

    command = method.infer(
        CanonicalFrame(np.zeros((21, 3)), 3.5, "right")
    )

    assert command.joint_names == ("joint_b", "joint_a")
    np.testing.assert_array_equal(
        command.qpos, np.array([0.2, -0.1], dtype=np.float32)
    )
    assert command.timestamp == 3.5


class _Retargeter:
    joint_names = ["joint_b", "joint_a"]
    optimizer = SimpleNamespace(
        target_link_human_indices=np.array([[0, 0], [4, 8]])
    )

    def retarget(self, reference):
        self.reference = reference.copy()
        return np.array([2.0, 1.0])


def test_dexpilot_maps_its_output_to_robot_joint_order():
    retargeter = _Retargeter()
    method = DexPilotMethod(
        {"joint_order": ["joint_a", "joint_b"]}, retargeter=retargeter
    )
    points = np.zeros((21, 3), dtype=np.float32)
    points[4, 0] = 0.1
    points[8, 1] = 0.2

    command = method.infer(CanonicalFrame(points, 7.0, "right"))

    np.testing.assert_allclose(retargeter.reference, [[0.1, 0, 0], [0, 0.2, 0]])
    np.testing.assert_array_equal(command.qpos, [1.0, 2.0])
