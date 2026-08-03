import numpy as np

from experiments.schema import canonicalize_landmarks


WEBXR_TO_LANDMARK = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
)


def from_webxr(observation, timestamp, hand_side, calibration):
    observation = np.asarray(observation)
    if observation.shape == (25, 4, 4):
        positions = observation[:, :3, 3]
    elif observation.shape == (25, 3):
        positions = observation
    else:
        raise ValueError("WebXR observation must contain 25 positions or transforms")

    return canonicalize_landmarks(
        positions[list(WEBXR_TO_LANDMARK)],
        timestamp,
        hand_side,
        scale=float(calibration["scale"]),
        rotation=np.asarray(calibration["rotation"]),
        outward_sign=int(calibration["outward_sign"]),
    )
