from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robots.wuji.ros import (
    build_joint_trajectory,
    message_timestamp,
    pose_array_to_positions,
    retarget_pose_array,
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


def test_retarget_pose_array_only_returns_valid_runtime_commands():
    calibration = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "outward_sign": 1,
    }
    positions = np.zeros((25, 3))
    positions[11] = [0, 0, 0.1]
    positions[2] = [0, 0.03, 0]
    positions[21] = [0, -0.03, 0]

    def message(values):
        poses = [
            SimpleNamespace(
                position=SimpleNamespace(x=value[0], y=value[1], z=value[2])
            )
            for value in values
        ]
        stamp = SimpleNamespace(sec=1, nanosec=0)
        return SimpleNamespace(poses=poses, header=SimpleNamespace(stamp=stamp))

    class Runtime:
        def __init__(self):
            self.frame = None

        def infer(self, frame):
            self.frame = frame
            return "command" if frame.valid else None

    runtime = Runtime()
    assert retarget_pose_array(
        message(positions), "right", calibration, runtime
    ) == "command"
    assert runtime.frame.valid
    assert retarget_pose_array(
        message(np.zeros((25, 3))), "right", calibration, runtime
    ) is None
