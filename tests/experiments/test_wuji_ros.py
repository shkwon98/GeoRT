from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robots.wuji.ros import (
    build_joint_trajectory,
    message_timestamp,
    pose_array_to_positions,
)
from experiments.schema import NamedCommand


class Trajectory:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.joint_names = []
        self.points = []


class Point:
    def __init__(self):
        self.positions = []
        self.time_from_start = SimpleNamespace(sec=-1, nanosec=-1)


def test_pose_array_and_trajectory_use_existing_wuji_contract():
    poses = [
        SimpleNamespace(position=SimpleNamespace(x=i, y=i + 1, z=i + 2))
        for i in range(25)
    ]
    stamp = SimpleNamespace(sec=4, nanosec=500_000_000)
    message = SimpleNamespace(poses=poses, header=SimpleNamespace(stamp=stamp))
    positions = pose_array_to_positions(message)

    assert positions.shape == (25, 3)
    assert message_timestamp(message) == 4.5

    command = NamedCommand(("right_a", "right_b"), np.array([0.1, 0.2]), 4.5)
    trajectory = build_joint_trajectory(
        command, "right", stamp, message_types=(Trajectory, Point)
    )
    assert trajectory.header.frame_id == "right_hand"
    assert trajectory.header.stamp is stamp
    assert trajectory.joint_names == ["right_a", "right_b"]
    assert trajectory.points[0].positions == pytest.approx([0.1, 0.2])
    assert trajectory.points[0].time_from_start.sec == 0
    assert trajectory.points[0].time_from_start.nanosec == 0


def test_pose_array_requires_all_25_webxr_joints():
    message = SimpleNamespace(poses=[])
    with pytest.raises(ValueError, match="25"):
        pose_array_to_positions(message)


def test_trajectory_preserves_all_20_wuji_joint_names():
    names = tuple(
        f"right_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    )
    command = NamedCommand(names, np.zeros(20), 0.0)
    trajectory = build_joint_trajectory(
        command, "right", None, message_types=(Trajectory, Point)
    )

    assert trajectory.joint_names == list(names)
    assert len(trajectory.points[0].positions) == 20
