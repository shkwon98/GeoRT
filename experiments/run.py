import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.artifacts import (
    create_run_dir,
    save_canonical_recording,
    save_commands,
    save_raw_recording,
)
from experiments.evaluate import evaluate_trajectory
from experiments.mocap.manus import from_manus
from experiments.mocap.mediapipe import from_mediapipe
from experiments.mocap.replay import load_replay
from experiments.mocap.webxr import from_webxr
from experiments.robots import load_robot_spec
from experiments.schema import validate_calibration


_CONFIG_FIELDS = {
    "mocap",
    "method",
    "robot",
    "hand_side",
    "calibration",
    "method_options",
}


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--input", required=True)
    collect.add_argument("--timestamps")
    collect.add_argument("--run-id")

    train = commands.add_parser("train")
    train.add_argument("--run-dir", required=True)

    export = commands.add_parser("export")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--checkpoint")

    infer = commands.add_parser("infer")
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--checkpoint")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", required=True)
    return parser


def load_experiment(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != _CONFIG_FIELDS:
        raise ValueError(f"experiment fields must be exactly {sorted(_CONFIG_FIELDS)}")
    if config["mocap"] not in {"webxr", "manus", "mediapipe", "replay"}:
        raise ValueError("unsupported mocap")
    if config["method"] not in {"geort", "dexpilot"}:
        raise ValueError("unsupported method")
    if not isinstance(config["method_options"], dict):
        raise ValueError("method_options must be an object")
    scale, rotation, outward_sign = validate_calibration(
        config["calibration"].get("scale"),
        config["calibration"].get("rotation"),
        config["calibration"].get("outward_sign"),
    )
    robot = load_robot_spec(config["robot"])
    if config["hand_side"] != robot["hand_side"]:
        raise ValueError("experiment hand_side must match the robot")
    config["calibration"] = {
        "scale": scale,
        "rotation": rotation.tolist(),
        "outward_sign": outward_sign,
    }
    config["robot_spec"] = robot
    return config


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _versions():
    try:
        dexpilot = importlib.metadata.version("dex-retargeting")
    except importlib.metadata.PackageNotFoundError:
        dexpilot = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dex-retargeting": dexpilot,
    }


def _collect(args):
    resolved = load_experiment(args.config)
    robot = resolved.pop("robot_spec")
    run_dir = create_run_dir(args.run_id)
    _write_json(run_dir / "config.json", resolved)
    _write_json(run_dir / "robot.json", robot)
    _write_json(run_dir / "versions.json", _versions())

    if resolved["mocap"] == "replay":
        metadata_path = (
            Path(args.timestamps)
            if args.timestamps
            else Path(args.input).with_name("canonical_metadata.json")
        )
        frames = load_replay(args.input, metadata_path)
        observations = np.load(args.input)
        timestamps = np.array([frame.timestamp for frame in frames])
    else:
        observations = np.load(args.input)
        if observations.ndim < 1:
            raise ValueError("raw observations must contain frames")
        timestamps = (
            np.load(args.timestamps)
            if args.timestamps
            else np.arange(len(observations), dtype=np.float64)
        )
        adapter = {
            "webxr": from_webxr,
            "manus": from_manus,
            "mediapipe": from_mediapipe,
        }[resolved["mocap"]]
        frames = [
            adapter(
                observation,
                timestamp,
                resolved["hand_side"],
                resolved["calibration"],
            )
            for observation, timestamp in zip(observations, timestamps)
        ]

    save_raw_recording(
        run_dir,
        observations,
        timestamps,
        {
            "mocap": resolved["mocap"],
            "hand_side": resolved["hand_side"],
            "units": "m",
            "source_frame": resolved["mocap"],
            "calibration": resolved["calibration"],
        },
    )
    save_canonical_recording(
        run_dir,
        frames,
        {"mocap": resolved["mocap"], "calibration": resolved["calibration"]},
    )
    print(run_dir)
    return run_dir


def _load_run(run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    robot_path = run_dir / "robot.json"
    if not config_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError("run requires config.json and robot.json")
    return (
        run_dir,
        json.loads(config_path.read_text(encoding="utf-8")),
        json.loads(robot_path.read_text(encoding="utf-8")),
    )


def _validate_robot_snapshot(robot):
    path = Path(robot["urdf_path"])
    if not path.is_file():
        raise FileNotFoundError(f"robot URDF not found: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != robot.get("urdf_sha256"):
        raise ValueError("robot URDF changed after experiment collection")


def _checkpoint(run_dir, explicit=None):
    if explicit:
        path = Path(explicit)
    else:
        training_path = run_dir / "training.json"
        if not training_path.is_file():
            raise FileNotFoundError("checkpoint is required before GeoRT inference")
        path = Path(
            json.loads(training_path.read_text(encoding="utf-8"))["checkpoint"]
        )
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")
    return path.resolve()


def _train(args):
    run_dir, config, robot = _load_run(args.run_dir)
    _validate_robot_snapshot(robot)
    if config["method"] != "geort":
        raise ValueError("DexPilot is an online baseline and is not trained")
    canonical = run_dir / "canonical.npy"
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    from experiments.methods.geort import train

    checkpoint = train(
        canonical, robot, run_dir, config.get("method_options", {})
    ).resolve()
    _write_json(
        run_dir / "training.json",
        {"checkpoint": str(checkpoint), "options": config["method_options"]},
    )
    return checkpoint


def _export(args):
    run_dir, config, robot = _load_run(args.run_dir)
    _validate_robot_snapshot(robot)
    if config["method"] != "geort":
        raise ValueError("only GeoRT models can be exported")
    checkpoint = _checkpoint(run_dir, args.checkpoint)
    from experiments.methods.geort import export_torchscript

    return export_torchscript(
        checkpoint, robot, run_dir / "model.ts", config, device="cpu"
    )


def _make_method(config, robot, checkpoint=None):
    if config["method"] == "geort":
        from experiments.methods.geort import GeoRTMethod

        return GeoRTMethod(
            checkpoint, robot, device=config.get("method_options", {}).get("device")
        )
    from experiments.methods.dexpilot import DexPilotMethod

    return DexPilotMethod(robot, options=config.get("method_options"))


def _infer(args):
    run_dir, config, robot = _load_run(args.run_dir)
    _validate_robot_snapshot(robot)
    frames = load_replay(
        run_dir / "canonical.npy", run_dir / "canonical_metadata.json"
    )
    checkpoint = (
        _checkpoint(run_dir, args.checkpoint)
        if config["method"] == "geort"
        else None
    )
    method = _make_method(config, robot, checkpoint)
    commands = []
    latencies = []
    for frame in frames:
        if not frame.valid:
            continue
        start = time.perf_counter()
        commands.append(method.infer(frame))
        latencies.append(time.perf_counter() - start)
    save_commands(run_dir, commands)
    np.save(run_dir / "latency.npy", np.asarray(latencies, dtype=np.float64))
    return run_dir / "qpos.npz"


def _build_hand(robot):
    from geort.env.hand import HandKinematicModel

    return HandKinematicModel.build_from_config(robot)


def _evaluate(args):
    run_dir, _, robot = _load_run(args.run_dir)
    _validate_robot_snapshot(robot)
    frames = [
        frame
        for frame in load_replay(
            run_dir / "canonical.npy", run_dir / "canonical_metadata.json"
        )
        if frame.valid
    ]
    latency_path = run_dir / "latency.npy"
    command_path = run_dir / "qpos.npz"
    if not latency_path.is_file() or not command_path.is_file():
        raise FileNotFoundError("infer must run before evaluate")
    with np.load(command_path) as commands:
        qpos = commands["qpos"]
        joint_names = tuple(commands["joint_names"].tolist())
        timestamps = commands["timestamps"]
    if len(frames) != len(qpos) or not np.allclose(
        timestamps, [frame.timestamp for frame in frames], rtol=0, atol=0
    ):
        raise ValueError("commands do not align with canonical frames")
    metrics = evaluate_trajectory(
        frames,
        qpos,
        joint_names,
        np.load(latency_path),
        robot,
        _build_hand(robot),
    )
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {
        "collect": _collect,
        "train": _train,
        "export": _export,
        "infer": _infer,
        "evaluate": _evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
