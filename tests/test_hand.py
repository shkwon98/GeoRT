import numpy as np
import pytest

pytest.importorskip("sapien")
from geort.env.hand import HandKinematicModel


class Joint:
    def __init__(self):
        self.target = None

    def set_drive_target(self, target):
        self.target = target


def _hand_model():
    model = HandKinematicModel.__new__(HandKinematicModel)
    model.joint_lower_limit = np.array([-10.0, -10.0, -10.0])
    model.joint_upper_limit = np.array([10.0, 10.0, 10.0])
    model.sim_idx_to_user_idx = [1, 2, 0]
    model.all_joints = [Joint(), Joint(), Joint()]
    return model


def test_set_qpos_target_keeps_user_targets_with_user_named_joints():
    model = _hand_model()

    model.set_qpos_target(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(model.qpos_target, [2.0, 3.0, 1.0])
    assert [joint.target for joint in model.all_joints] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "qpos",
    [np.array([1.0, 2.0]), np.array([1.0, np.nan, 3.0])],
)
def test_set_qpos_target_rejects_invalid_qpos(qpos):
    with pytest.raises(ValueError, match="qpos"):
        _hand_model().set_qpos_target(qpos)
