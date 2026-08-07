import numpy as np
import pytest

from geort.schema import NamedCommand, canonicalize_landmarks, validate_calibration


def test_canonicalize_landmarks_builds_wrist_relative_axes():
    points = np.zeros((21, 3), dtype=np.float64)
    points[9] = [0, 0, 2]
    points[2] = [0, 2, 0]
    points[17] = [0, -1, 0]

    frame = canonicalize_landmarks(
        points, 1.25, "right", 0.5, np.eye(3), 1
    )

    np.testing.assert_allclose(frame.points[0], 0)
    np.testing.assert_allclose(frame.points[9], [0, 0, 1])
    assert frame.points.dtype == np.float32
    assert frame.valid


def test_schema_rejects_bad_commands_and_degenerate_calibration():
    invalid = canonicalize_landmarks(
        np.zeros((21, 3)), 0.0, "left", 1.0, np.eye(3), -1
    )

    assert not invalid.valid
    with pytest.raises(ValueError, match="joint_names"):
        NamedCommand(("a", "a"), np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="qpos"):
        NamedCommand(("a",), np.array([np.nan]), 0.0)
    with pytest.raises(ValueError, match="rotation"):
        validate_calibration(1.0, np.zeros((3, 3)), 1)
