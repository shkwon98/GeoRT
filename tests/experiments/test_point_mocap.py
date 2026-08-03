import numpy as np
import pytest

from experiments.mocap.manus import from_manus
from experiments.mocap.mediapipe import from_mediapipe


def _points():
    points = np.zeros((21, 3))
    points[9] = [0, 0, 0.1]
    points[2] = [0, 0.03, 0]
    points[17] = [0, -0.03, 0]
    return points


def test_manus_and_mediapipe_world_points_share_the_canonical_contract():
    calibration = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "outward_sign": 1,
    }
    manus = from_manus(_points(), 1.0, "right", calibration)
    mediapipe = from_mediapipe(
        _points(), 1.0, "right", calibration, world=True
    )

    np.testing.assert_allclose(manus.points, mediapipe.points)
    assert manus.points.dtype == np.float32


def test_mediapipe_normalized_landmarks_require_explicit_demo_mode():
    calibration = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "outward_sign": 1,
    }
    with pytest.raises(ValueError, match="world landmarks"):
        from_mediapipe(_points(), 1.0, "right", calibration, world=False)
