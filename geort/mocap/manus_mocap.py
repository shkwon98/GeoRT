# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from geort.mocap.manus_kinematics import manus_keypoints


class ManusMocap:
    def __init__(self, topic="/manus_quats"):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Float32MultiArray

        self._context = Context()
        rclpy.init(context=self._context)
        self._rclpy = rclpy
        self._node = rclpy.create_node(
            "geort_manus_mocap", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._subscription = self._node.create_subscription(
            Float32MultiArray,
            topic,
            self._on_quaternions,
            10,
        )
        self._latest_data = None
        self._new_data = False
        self._closed = False

    def _on_quaternions(self, message):
        try:
            quaternions = np.asarray(message.data).reshape(21, 4)
            self._latest_data = manus_keypoints(quaternions)
            self._new_data = True
        except ValueError as error:
            self._node.get_logger().warning(str(error))

    def get(self):
        self._executor.spin_once(timeout_sec=0.0)
        if self._latest_data is None or not self._new_data:
            return {"result": None, "status": "no data"}
        self._new_data = False
        return {"result": self._latest_data.copy(), "status": "recording"}

    def close(self):
        if self._closed:
            return
        self._executor.remove_node(self._node)
        self._executor.shutdown()
        self._node.destroy_node()
        self._rclpy.shutdown(context=self._context)
        self._closed = True
