import json
from pathlib import Path

import numpy as np

from experiments.schema import CanonicalFrame


def load_replay(points_path, metadata_path):
    points = np.load(Path(points_path))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if points.ndim != 3 or points.shape[1:] != (21, 3):
        raise ValueError("canonical points must have shape (T, 21, 3)")

    timestamps = np.asarray(metadata.get("timestamps"), dtype=np.float64)
    validity = metadata.get("valid")
    hand_side = metadata.get("hand_side")
    if (
        timestamps.shape != (len(points),)
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) < 0)
    ):
        raise ValueError("timestamps must be finite, monotonic, and match frames")
    if not isinstance(validity, list) or len(validity) != len(points):
        raise ValueError("valid must match canonical frames")
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")

    return [
        CanonicalFrame(point, timestamp, hand_side, bool(valid))
        for point, timestamp, valid in zip(points, timestamps, validity)
    ]
