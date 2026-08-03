from types import SimpleNamespace

import numpy as np

from experiments.methods.dexpilot import DexPilotMethod
from experiments.schema import CanonicalFrame


class Retargeter:
    joint_names = ["joint_b", "joint_a"]
    optimizer = SimpleNamespace(
        target_link_human_indices=np.array([[0, 0], [4, 8]])
    )

    def __init__(self):
        self.reference = None

    def retarget(self, reference):
        self.reference = reference.copy()
        return np.array([2.0, 1.0])


def test_dexpilot_builds_reference_vectors_and_maps_output_by_name():
    retargeter = Retargeter()
    method = DexPilotMethod(
        {"joint_order": ["joint_a", "joint_b"]}, retargeter=retargeter
    )
    points = np.zeros((21, 3), dtype=np.float32)
    points[4, 0] = 0.1
    points[8, 1] = 0.2

    command = method.infer(CanonicalFrame(points, 7.0, "right"))

    np.testing.assert_allclose(
        retargeter.reference, [[0.1, 0, 0], [0, 0.2, 0]]
    )
    assert command.joint_names == ("joint_a", "joint_b")
    np.testing.assert_array_equal(command.qpos, [1.0, 2.0])
