import numpy as np
import pytest

from geort.dataset import GestureDataset, upsample_array


def test_upsample_array_handles_boundary_inputs():
    point = np.array([[1.0, 2.0, 3.0]])

    np.testing.assert_array_equal(upsample_array(point, K=4), np.repeat(point, 4, axis=0))
    with pytest.raises(ValueError, match="empty"):
        upsample_array(np.empty((0, 3)), K=4)


def test_gesture_dataset_keeps_frames_aligned():
    frames = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    np.testing.assert_array_equal(GestureDataset(frames)[1], frames[1])
