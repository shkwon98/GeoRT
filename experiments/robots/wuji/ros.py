import numpy as np
from pathlib import Path

from experiments.mocap.webxr import from_webxr


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


def retarget_pose_array(message, side, calibration, runtime):
    frame = from_webxr(
        pose_array_to_positions(message),
        message_timestamp(message),
        side,
        calibration,
    )
    return runtime.infer(frame)


def create_node_class():
    from geometry_msgs.msg import PoseArray
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from trajectory_msgs.msg import JointTrajectory

    from experiments.robots import load_robot_spec
    from experiments.robots.wuji.runtime import TorchScriptRuntime

    class WujiGeoRTRollout(Node):
        def __init__(self):
            super().__init__("geort_wuji_rollout")
            self.declare_parameter("model", "")
            self.declare_parameter("metadata", "")
            self.declare_parameter("side", "right")
            self.declare_parameter("alpha", 0.2)

            model_value = self.get_parameter("model").value
            if not model_value:
                raise ValueError("model parameter is required")
            model_path = Path(model_value).expanduser()
            metadata_value = self.get_parameter("metadata").value
            metadata_path = (
                Path(metadata_value).expanduser()
                if metadata_value
                else model_path.with_suffix(".json")
            )
            side = self.get_parameter("side").value
            if side not in {"left", "right"}:
                raise ValueError("side must be 'left' or 'right'")

            robot = load_robot_spec(f"wuji_{side}")
            self._runtime = TorchScriptRuntime(
                model_path,
                metadata_path,
                robot,
                alpha=self.get_parameter("alpha").value,
            )
            self._side = side
            self._calibration = self._runtime.metadata["calibration"]
            self.declare_parameter("input_topic", robot["ros"]["input_topic"])
            self.declare_parameter("output_topic", robot["ros"]["output_topic"])
            input_topic = self.get_parameter("input_topic").value
            output_topic = self.get_parameter("output_topic").value

            input_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            output_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self._publisher = self.create_publisher(
                JointTrajectory, output_topic, output_qos
            )
            self._subscription = self.create_subscription(
                PoseArray, input_topic, self._on_pose_array, input_qos
            )
            self.get_logger().info(
                f"GeoRT Wuji {side}: {input_topic} -> {output_topic}; "
                f"joints={list(self._runtime.joint_names)}"
            )

        def _on_pose_array(self, message):
            try:
                command = retarget_pose_array(
                    message,
                    self._side,
                    self._calibration,
                    self._runtime,
                )
                if command is None:
                    return
                self._publisher.publish(
                    build_joint_trajectory(
                        command, self._side, message.header.stamp
                    )
                )
            except Exception as error:
                self.get_logger().warning(
                    f"Dropped invalid hand frame: {error}",
                    throttle_duration_sec=5.0,
                )

    return WujiGeoRTRollout


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = create_node_class()()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
