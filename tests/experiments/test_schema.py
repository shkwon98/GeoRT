import numpy as np
import pytest

from experiments.schema import (
    CanonicalFrame,
    NamedCommand,
    canonicalize_landmarks,
    validate_calibration,
)


def _hand_points():
    points = np.zeros((21, 3), dtype=np.float64)
    points[9] = [0.0, 0.0, 2.0]
    points[2] = [0.0, 2.0, 0.0]
    points[17] = [0.0, -1.0, 0.0]
    points[4] = [1.0, 3.0, 1.0]
    return points


def test_canonicalize_landmarks_builds_wrist_relative_orthogonal_axes():
    frame = canonicalize_landmarks(
        _hand_points(),
        timestamp=1.25,
        hand_side="right",
        scale=0.5,
        rotation=np.eye(3),
        outward_sign=1,
    )

    assert frame.points.shape == (21, 3)
    assert frame.points.dtype == np.float32
    np.testing.assert_allclose(frame.points[0], 0.0)
    np.testing.assert_allclose(frame.points[9], [0.0, 0.0, 1.0])
    assert frame.timestamp == 1.25
    assert frame.hand_side == "right"
    assert frame.valid


def test_schema_rejects_bad_commands_and_marks_degenerate_frames_invalid():
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
