import numpy as np
import pytest

from experiments.mocap.webxr import WEBXR_TO_LANDMARK, from_webxr


EXPECTED = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)


def test_webxr_maps_25_joints_and_accepts_transform_matrices():
    positions = np.zeros((25, 3), dtype=np.float64)
    positions[:, 0] = np.arange(25)
    positions[14] = [0.0, 0.0, 2.0]
    positions[11] = [0.0, 0.0, 1.0]
    positions[2] = [0.0, 1.0, 0.0]
    positions[21] = [0.0, -1.0, 0.0]
    transforms = np.repeat(np.eye(4)[None], 25, axis=0)
    transforms[:, :3, 3] = positions
    calibration = {
        "scale": 1.0,
        "rotation": np.eye(3),
        "outward_sign": 1,
    }

    frame = from_webxr(transforms, 2.0, "right", calibration)

    assert WEBXR_TO_LANDMARK == EXPECTED
    assert frame.valid
    assert frame.points.shape == (21, 3)
    with pytest.raises(ValueError, match="25"):
        from_webxr(transforms[:24], 2.0, "right", calibration)
