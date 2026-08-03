import numpy as np

from experiments.schema import canonicalize_landmarks


def from_mediapipe(points, timestamp, hand_side, calibration, world=True):
    if not world:
        raise ValueError("MediaPipe training requires metric world landmarks")
    return canonicalize_landmarks(
        points,
        timestamp,
        hand_side,
        scale=float(calibration["scale"]),
        rotation=np.asarray(calibration["rotation"]),
        outward_sign=int(calibration["outward_sign"]),
    )
