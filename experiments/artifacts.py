import json
from datetime import datetime
from pathlib import Path

import numpy as np

from geort.utils.path import get_run_root


def create_run_dir(run_id=None, root=None):
    root = (
        Path(root)
        if root is not None
        else get_run_root()
    )
    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _ensure_new(*paths):
    for path in paths:
        if path.exists():
            raise FileExistsError(path)


def _timestamps(values, frame_count):
    timestamps = np.asarray(values, dtype=np.float64)
    if (
        timestamps.shape != (frame_count,)
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) < 0)
    ):
        raise ValueError(
            "timestamps must be finite, monotonic, and match frames")
    return timestamps


def _write_json(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(encoded)
        file.write("\n")


def save_raw_recording(run_dir, observations, timestamps, metadata):
    run_dir = Path(run_dir)
    data_path = run_dir / "raw.npz"
    metadata_path = run_dir / "raw_metadata.json"
    _ensure_new(data_path, metadata_path)

    required = {"mocap", "hand_side", "units", "source_frame", "calibration"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"raw metadata is missing: {sorted(missing)}")
    observations = np.asarray(observations)
    if observations.ndim < 1:
        raise ValueError("observations must contain a frame dimension")
    timestamps = _timestamps(timestamps, len(observations))
    payload = dict(metadata)
    payload["timestamps"] = timestamps.tolist()
    json.dumps(payload)

    with data_path.open("xb") as file:
        np.savez(file, observations=observations, timestamps=timestamps)
    _write_json(metadata_path, payload)
    return data_path


def save_canonical_recording(run_dir, frames, metadata):
    run_dir = Path(run_dir)
    data_path = run_dir / "canonical.npy"
    metadata_path = run_dir / "canonical_metadata.json"
    _ensure_new(data_path, metadata_path)
    frames = list(frames)
    if not frames:
        raise ValueError("canonical recording must contain frames")
    hand_sides = {frame.hand_side for frame in frames}
    if len(hand_sides) != 1:
        raise ValueError("canonical frames must share one hand_side")
    timestamps = _timestamps(
        [frame.timestamp for frame in frames], len(frames))
    points = np.stack([frame.points for frame in frames]).astype(np.float32)
    payload = {
        **metadata,
        "hand_side": hand_sides.pop(),
        "timestamps": timestamps.tolist(),
        "valid": [bool(frame.valid) for frame in frames],
    }
    json.dumps(payload)

    with data_path.open("xb") as file:
        np.save(file, points)
    _write_json(metadata_path, payload)
    return data_path


def save_commands(run_dir, commands):
    path = Path(run_dir) / "qpos.npz"
    _ensure_new(path)
    commands = list(commands)
    if not commands:
        raise ValueError("commands must not be empty")
    joint_names = commands[0].joint_names
    if any(command.joint_names != joint_names for command in commands[1:]):
        raise ValueError("commands must share joint_names")
    timestamps = _timestamps(
        [command.timestamp for command in commands], len(commands)
    )
    qpos = np.stack([command.qpos for command in commands]).astype(np.float32)

    with path.open("xb") as file:
        np.savez(
            file,
            joint_names=np.asarray(joint_names),
            qpos=qpos,
            timestamps=timestamps,
        )
    return path
