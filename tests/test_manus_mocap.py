import numpy as np
import pytest

from geort.mocap.manus_kinematics import manus_keypoints


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
