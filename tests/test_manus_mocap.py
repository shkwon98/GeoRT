from types import SimpleNamespace

import numpy as np
import pytest

from geort.mocap.manus_kinematics import manus_keypoints
from geort.mocap.manus_mocap import ManusMocap


def identity_quaternions():
    quaternions = np.zeros((21, 4), dtype=np.float64)
    quaternions[:, 3] = 1.0
    return quaternions


def test_manus_keypoints_converts_identity_quaternions():
    points = manus_keypoints(identity_quaternions())

    assert points.shape == (21, 3)
    assert points.dtype == np.float32
    np.testing.assert_allclose(points[0], 0.0, atol=1e-7)
    assert np.isfinite(points).all()


@pytest.mark.parametrize(
    "quaternions",
    [np.zeros((20, 4)), np.full((21, 4), np.nan)],
)
def test_manus_keypoints_rejects_invalid_quaternions(quaternions):
    with pytest.raises(ValueError):
        manus_keypoints(quaternions)


class Executor:
    def __init__(self):
        self.calls = 0

    def spin_once(self, timeout_sec):
        assert timeout_sec == 0.0
        self.calls += 1


def test_manus_mocap_polls_ros_and_returns_copied_keypoints():
    mocap = ManusMocap.__new__(ManusMocap)
    mocap._executor = Executor()
    mocap._latest_data = None

    assert mocap.get() == {"result": None, "status": "no data"}

    message = SimpleNamespace(data=identity_quaternions().reshape(-1))
    mocap._on_quaternions(message)
    first = mocap.get()
    first["result"][0, 0] = 1.0
    second = mocap.get()

    assert first["status"] == "recording"
    assert second["result"].shape == (21, 3)
    assert second["result"][0, 0] == 0.0
    assert mocap._executor.calls == 3
