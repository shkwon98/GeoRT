from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalFrame:
    points: np.ndarray
    timestamp: float
    hand_side: str
    valid: bool = True

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float32)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("points must be finite with shape (21, 3)")
        if self.hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be 'left' or 'right'")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class NamedCommand:
    joint_names: tuple[str, ...]
    qpos: np.ndarray
    timestamp: float

    def __post_init__(self):
        names = tuple(self.joint_names)
        qpos = np.asarray(self.qpos, dtype=np.float32)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint_names must be non-empty and unique")
        if qpos.shape != (len(names),) or not np.isfinite(qpos).all():
            raise ValueError("qpos must be finite and match joint_names")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "timestamp", float(self.timestamp))


def validate_calibration(scale, rotation, outward_sign):
    scale = float(scale)
    rotation = np.asarray(rotation, dtype=np.float64)
    outward_sign = int(outward_sign)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    ):
        raise ValueError("rotation must be a finite orthogonal 3x3 matrix")
    if outward_sign not in {-1, 1}:
        raise ValueError("outward_sign must be -1 or 1")
    return scale, rotation, outward_sign


def canonicalize_landmarks(
    points,
    timestamp,
    hand_side,
    scale,
    rotation,
    outward_sign,
):
    scale, rotation, outward_sign = validate_calibration(
        scale, rotation, outward_sign
    )
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape (21, 3)")

    centered = (points @ rotation.T) * scale
    centered -= centered[0]
    z_axis = centered[9]
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return CanonicalFrame(np.zeros((21, 3)), timestamp, hand_side, False)
    z_axis /= z_norm

    y_axis = centered[2] - centered[17]
    y_axis -= np.dot(y_axis, z_axis) * z_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return CanonicalFrame(np.zeros((21, 3)), timestamp, hand_side, False)
    y_axis /= y_norm
    x_axis = outward_sign * np.cross(y_axis, z_axis)
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)
    return CanonicalFrame(centered @ basis, timestamp, hand_side)
