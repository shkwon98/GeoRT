import numpy as np
import pytest
import sys
from types import SimpleNamespace

from torch.utils.data import Dataset

from geort.dataset import (
    GestureDataset,
    MultiPointDataset,
    RobotKinematicsDataset,
    upsample_array,
)


def test_upsample_array_handles_boundary_inputs():
    point = np.array([[1.0, 2.0, 3.0]])

    np.testing.assert_array_equal(upsample_array(
        point, K=4), np.repeat(point, 4, axis=0))
    with pytest.raises(ValueError, match="empty"):
        upsample_array(np.empty((0, 3)), K=4)


def test_gesture_dataset_keeps_frames_aligned():
    frames = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    np.testing.assert_array_equal(GestureDataset(frames)[1], frames[1])


def test_from_points_honors_requested_sample_count(monkeypatch):
    class PointCloud:
        def __init__(self):
            self.points = None

        def voxel_down_sample(self, voxel_size):
            return self

    fake_open3d = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=PointCloud),
        utility=SimpleNamespace(Vector3dVector=np.asarray),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)
    points = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    dataset = MultiPointDataset.from_points(points, n=7)

    assert dataset.points.shape == (2, 7, 3)
    with pytest.raises(ValueError, match="n must be positive"):
        MultiPointDataset.from_points(points, n=0)


def test_robot_kinematics_dataset_uses_torch_dataset_protocol(tmp_path):
    path = tmp_path / "kinematics.npz"
    np.savez(
        path,
        qpos=np.array([[1.0, 2.0], [3.0, 4.0]]),
        keypoint=np.array(
            {"tip": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            dtype=object,
        ),
    )

    dataset = RobotKinematicsDataset(path, ["tip"])

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
