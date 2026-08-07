from pathlib import Path

import numpy as np

from geort.schema import CanonicalFrame, validate_calibration
from geort.utils.config_utils import load_json, save_json
from geort.utils.path import get_data_root, get_run_root


def _id(value, kind):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{kind} must be a single path component")
    return value


def create_dataset(dataset_id, observations, frames, metadata, root=None):
    dataset_id = _id(dataset_id, "dataset id")
    observations = np.asarray(observations)
    frames = list(frames)
    if observations.ndim < 1 or len(observations) != len(frames) or not frames:
        raise ValueError("observations and canonical frames must be non-empty and aligned")
    hand_sides = {frame.hand_side for frame in frames}
    if hand_sides != {metadata.get("hand_side")}:
        raise ValueError("metadata hand_side must match canonical frames")
    calibration = metadata.get("calibration", {})
    scale, rotation, outward_sign = validate_calibration(
        calibration.get("scale"),
        calibration.get("rotation"),
        calibration.get("outward_sign"),
    )
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be monotonic")
    points = np.stack([frame.points for frame in frames]).astype(np.float32)
    payload = {
        **metadata,
        "dataset": dataset_id,
        "calibration": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "outward_sign": outward_sign,
        },
        "timestamps": timestamps.tolist(),
        "valid": [bool(frame.valid) for frame in frames],
    }

    parent = Path(root) / "data" if root is not None else get_data_root()
    dataset_dir = parent / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=False)
    np.savez(dataset_dir / "raw.npz", observations=observations, timestamps=timestamps)
    np.save(dataset_dir / "canonical.npy", points)
    save_json(payload, dataset_dir / "metadata.json")
    return dataset_dir


def create_run(run_id, config, root=None):
    run_id = _id(run_id, "run id")
    required = {"dataset", "robot", "method", "seed"}
    if required - config.keys():
        raise ValueError(f"run config is missing: {sorted(required - config.keys())}")
    parent = Path(root) / "runs" if root is not None else get_run_root()
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "outputs").mkdir()
    save_json(config, run_dir / "config.json")
    return run_dir


def load_dataset(dataset_id, root=None):
    dataset_id = _id(dataset_id, "dataset id")
    parent = Path(root) / "data" if root is not None else get_data_root()
    dataset_dir = parent / dataset_id
    metadata = load_json(dataset_dir / "metadata.json")
    points = np.load(dataset_dir / "canonical.npy", allow_pickle=False)
    with np.load(dataset_dir / "raw.npz", allow_pickle=False) as raw:
        observations = raw["observations"]
        raw_timestamps = raw["timestamps"]
    timestamps = np.asarray(metadata.get("timestamps"), dtype=np.float64)
    validity = metadata.get("valid")
    if (
        points.shape != (len(observations), 21, 3)
        or timestamps.shape != (len(points),)
        or raw_timestamps.shape != timestamps.shape
        or not np.array_equal(raw_timestamps, timestamps)
        or not isinstance(validity, list)
        or len(validity) != len(points)
    ):
        raise ValueError(f"invalid dataset artifacts: {dataset_id}")
    frames = [
        CanonicalFrame(point, timestamp, metadata.get("hand_side"), bool(valid))
        for point, timestamp, valid in zip(points, timestamps, validity)
    ]
    return observations, frames, metadata


def load_run(run_id, root=None):
    run_id = _id(run_id, "run id")
    parent = Path(root) / "runs" if root is not None else get_run_root()
    run_dir = parent / run_id
    config = load_json(run_dir / "config.json")
    return run_dir, config


def save_commands(run_dir, commands, overwrite=False):
    commands = list(commands)
    if not commands:
        raise ValueError("commands must not be empty")
    joint_names = commands[0].joint_names
    if any(command.joint_names != joint_names for command in commands[1:]):
        raise ValueError("commands must share joint_names")
    timestamps = np.asarray([command.timestamp for command in commands])
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("command timestamps must be monotonic")
    output = Path(run_dir) / "outputs" / "qpos.npz"
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    np.savez(
        output,
        joint_names=np.asarray(joint_names),
        qpos=np.stack([command.qpos for command in commands]).astype(np.float32),
        timestamps=timestamps,
    )
    return output
