import numpy as np


class MetaQuestMocap:
    def __init__(self, topic):
        import rclpy
        from geometry_msgs.msg import PoseArray
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor

        self._context = Context()
        rclpy.init(context=self._context)
        self._rclpy = rclpy
        self._node = rclpy.create_node(
            "geort_metaquest_mocap", context=self._context
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._subscription = self._node.create_subscription(
            PoseArray, topic, self._on_poses, 10
        )
        self._latest_data = None
        self._latest_timestamp = None
        self._new_data = False
        self._closed = False

    def _on_poses(self, message):
        if len(message.poses) != 25:
            self._node.get_logger().warning(
                f"MetaQuest PoseArray must contain 25 poses, got {len(message.poses)}"
            )
            return
        self._latest_data = np.asarray(
            [
                [pose.position.x, pose.position.y, pose.position.z]
                for pose in message.poses
            ],
            dtype=np.float32,
        )
        self._latest_timestamp = (
            message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        )
        self._new_data = True

    def get(self):
        self._executor.spin_once(timeout_sec=0.0)
        if self._latest_data is None or not self._new_data:
            return {"result": None, "status": "no data"}
        self._new_data = False
        return {
            "result": self._latest_data.copy(),
            "status": "recording",
            "timestamp": self._latest_timestamp,
        }

    def close(self):
        if self._closed:
            return
        self._executor.remove_node(self._node)
        self._executor.shutdown()
        self._node.destroy_node()
        self._rclpy.shutdown(context=self._context)
        self._closed = True
