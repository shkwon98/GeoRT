import numpy as np
import pytest

from geort.mocap.adapters import METAQUEST_TO_LANDMARK, adapt_observation


def _points():
    points = np.zeros((21, 3))
    points[9] = [0, 0, 0.1]
    points[2] = [0, 0.03, 0]
    points[17] = [0, -0.03, 0]
    return points


def test_manus_and_mediapipe_use_the_same_metric_contract():
    calibration = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "outward_sign": 1,
    }
    manus = adapt_observation("manus", _points(), 1.0, "right", calibration)
    mediapipe = adapt_observation(
        "mediapipe", _points(), 1.0, "right", calibration
    )

    np.testing.assert_allclose(manus.points, mediapipe.points)
    assert manus.points.dtype == np.float32


def test_metaquest_maps_25_transforms_to_21_landmarks():
    positions = np.zeros((25, 3))
    positions[11] = [0, 0, 0.1]
    positions[2] = [0, 0.03, 0]
    positions[21] = [0, -0.03, 0]
    transforms = np.repeat(np.eye(4)[None], 25, axis=0)
    transforms[:, :3, 3] = positions
    calibration = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "outward_sign": 1,
    }

    frame = adapt_observation(
        "metaquest", transforms, 2.0, "right", calibration
    )

    assert len(METAQUEST_TO_LANDMARK) == 21
    assert frame.valid
    with pytest.raises(ValueError, match="25"):
        adapt_observation(
            "metaquest", transforms[:24], 2.0, "right", calibration
        )
