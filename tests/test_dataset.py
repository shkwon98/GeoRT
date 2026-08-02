import numpy as np
import pytest
from torch.utils.data import Dataset

from geort.dataset import GestureDataset, MultiPointDataset, upsample_array


def test_upsample_array_repeats_a_single_row():
    point = np.array([[1.0, 2.0, 3.0]])

    sampled = upsample_array(point, K=4)

    np.testing.assert_array_equal(sampled, np.repeat(point, 4, axis=0))


def test_upsample_array_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        upsample_array(np.empty((0, 3)), K=4)


def test_gesture_dataset_returns_frame_aligned_fingertips():
    frames = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    dataset = GestureDataset(frames)

    assert isinstance(dataset, Dataset)
    assert isinstance(MultiPointDataset(np.zeros((3, 2, 3))), Dataset)
    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset[1], frames[1])
