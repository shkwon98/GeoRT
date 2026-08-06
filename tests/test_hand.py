from types import SimpleNamespace

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")
from geort.env.hand import HandKinematicModel, HandViewerEnv
from geort.utils.config_utils import get_config


class Joint:
    def __init__(self):
        self.target = None

    def set_drive_target(self, target):
        self.target = target


class Hand:
    def set_qpos(self, qpos):
        self.qpos = qpos

    def set_qvel(self, qvel):
        self.qvel = qvel


def _hand_model():
    model = HandKinematicModel.__new__(HandKinematicModel)
    model.hand = Hand()
    model.joint_lower_limit = np.array([-10.0, -10.0, -10.0])
    model.joint_upper_limit = np.array([10.0, 10.0, 10.0])
    model.sim_idx_to_user_idx = [1, 2, 0]
    model.all_joints = [Joint(), Joint(), Joint()]
    return model


def test_set_qpos_target_keeps_user_targets_with_user_named_joints():
    model = _hand_model()

    model.set_qpos_target(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(model.qpos_target, [2.0, 3.0, 1.0])
    np.testing.assert_array_equal(model.hand.qpos, [2.0, 3.0, 1.0])
    np.testing.assert_array_equal(model.hand.qvel, [0.0, 0.0, 0.0])
    assert [joint.target for joint in model.all_joints] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "qpos",
    [np.array([1.0, 2.0]), np.array([1.0, np.nan, 3.0])],
)
def test_set_qpos_target_rejects_invalid_qpos(qpos):
    with pytest.raises(ValueError, match="qpos"):
        _hand_model().set_qpos_target(qpos)


def test_bundled_wuji_right_loads_as_a_20_dof_hand():
    model = HandKinematicModel.build_from_config(get_config("wuji_right"))

    assert model.get_n_dof() == 20


def test_wuji_collision_filter_distinguishes_open_and_closed_poses():
    model = HandKinematicModel.build_from_config(get_config("wuji_right"))

    assert not model.is_self_collision(model.joint_lower_limit + 1e-3)
    assert model.is_self_collision(model.joint_upper_limit - 1e-3)


def test_wuji_viewer_displays_targets_without_simulating_dynamics():
    model = HandKinematicModel.build_from_config(get_config("wuji_right"))
    target = model.joint_lower_limit + 0.25 * (
        model.joint_upper_limit - model.joint_lower_limit
    )
    viewer = HandViewerEnv.__new__(HandViewerEnv)
    viewer.model = model
    viewer.scene = model.scene
    viewer._mocap_inset = SimpleNamespace(set_view=lambda *args: None)
    viewer.viewer = SimpleNamespace(
        window=SimpleNamespace(get_camera_pose=sapien.Pose),
        render=lambda: None,
    )

    model.set_qpos_target(target)
    for _ in range(10):
        viewer.update()

    np.testing.assert_allclose(
        model.hand.get_qpos(),
        model.convert_user_order_to_sim_order(target),
        atol=1e-6,
    )
