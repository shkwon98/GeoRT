import numpy as np

from geort.schema import canonicalize_landmarks


METAQUEST_TO_LANDMARK = (
    0, 1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24,
)


def adapt_observation(mocap, observation, timestamp, hand_side, calibration):
    points = np.asarray(observation)
    if mocap == "metaquest":
        if points.shape == (25, 4, 4):
            points = points[:, :3, 3]
        elif points.shape != (25, 3):
            raise ValueError(
                "MetaQuest observation must contain 25 positions or transforms"
            )
        points = points[list(METAQUEST_TO_LANDMARK)]
    elif mocap not in {"manus", "mediapipe"}:
        raise ValueError(f"unsupported mocap: {mocap}")

    return canonicalize_landmarks(
        points,
        timestamp,
        hand_side,
        calibration.get("scale"),
        calibration.get("rotation"),
        calibration.get("outward_sign"),
    )
