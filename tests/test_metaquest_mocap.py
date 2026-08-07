from types import SimpleNamespace

import numpy as np

from geort.mocap.metaquest_mocap import MetaQuestMocap


class _Executor:
    def spin_once(self, timeout_sec):
        assert timeout_sec == 0.0


def test_metaquest_pose_array_preserves_positions_and_timestamp():
    mocap = MetaQuestMocap.__new__(MetaQuestMocap)
    mocap._executor = _Executor()
    mocap._latest_data = None
    mocap._latest_timestamp = None
    mocap._new_data = False
    poses = [
        SimpleNamespace(position=SimpleNamespace(x=index, y=1, z=2))
        for index in range(25)
    ]
    message = SimpleNamespace(
        poses=poses,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=500_000_000)),
    )

    mocap._on_poses(message)
    result = mocap.get()

    assert result["status"] == "recording"
    assert result["timestamp"] == 3.5
    np.testing.assert_array_equal(result["result"][:, 0], np.arange(25))
    assert mocap.get() == {"result": None, "status": "no data"}
