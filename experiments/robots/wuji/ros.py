import numpy as np


def pose_array_to_positions(message):
    if len(message.poses) < 25:
        raise ValueError("WebXR PoseArray must contain all 25 joints")
    positions = np.array(
        [
            [pose.position.x, pose.position.y, pose.position.z]
            for pose in message.poses[:25]
        ],
        dtype=np.float64,
    )
    if not np.isfinite(positions).all():
        raise ValueError("WebXR PoseArray positions must be finite")
    return positions


def message_timestamp(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def build_joint_trajectory(command, side, stamp, message_types=None):
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if message_types is None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    else:
        JointTrajectory, JointTrajectoryPoint = message_types

    trajectory = JointTrajectory()
    trajectory.header.frame_id = f"{side}_hand"
    if stamp is not None:
        trajectory.header.stamp = stamp
    trajectory.joint_names = list(command.joint_names)
    point = JointTrajectoryPoint()
    point.positions = command.qpos.tolist()
    point.time_from_start.sec = 0
    point.time_from_start.nanosec = 0
    trajectory.points.append(point)
    return trajectory
