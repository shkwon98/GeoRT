import argparse
import json
import time
from pathlib import Path

import numpy as np

from geort.artifacts import (
    create_dataset,
    create_run,
    load_dataset,
    load_run,
    save_commands,
)
from geort.metrics import evaluate_trajectory
from geort.methods import make_method
from geort.mocap.adapters import adapt_observation
from geort.robots import load_robot
from geort.schema import validate_calibration
from geort.utils.config_utils import load_json, save_json


_TRAIN_DEFAULTS = {
    "seed": 0,
    "save_every": 50,
    "val_fraction": 0.1,
    "coverage_samples": 20000,
    "coverage_batch_size": 2048,
    "gesture_batch_size": 2048,
}
_RESUME_SETTINGS = (
    *_TRAIN_DEFAULTS,
    "direction_sigma",
    "flatness_sigma",
    "w_chamfer",
    "w_curvature",
    "w_pinch",
    "w_collision",
)


def build_parser():
    parser = argparse.ArgumentParser(prog="geort")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect")
    collect.add_argument(
        "--mocap", choices=("manus", "mediapipe", "metaquest"), required=True
    )
    collect.add_argument("--dataset", required=True)
    collect.add_argument("--config", required=True)
    collect.add_argument("--input")
    collect.add_argument("--timestamps")

    train = commands.add_parser("train")
    train.add_argument("--dataset")
    train.add_argument("--robot")
    train.add_argument("--run-id")
    train.add_argument("--resume", metavar="RUN")
    train.add_argument("--device")
    train.add_argument("--seed", type=int)
    train.add_argument("--epoch", type=int)
    train.add_argument("--save-every", type=int)
    train.add_argument("--val-fraction", type=float)
    train.add_argument("--coverage-samples", type=int)
    train.add_argument("--coverage-batch-size", type=int)
    train.add_argument("--gesture-batch-size", type=int)

    rollout = commands.add_parser("rollout")
    rollout.add_argument("--run", required=True)
    rollout.add_argument("--source", choices=("live", "replay"), required=True)
    rollout.add_argument("--device")
    rollout.add_argument("--config")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run")
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--robot")
    evaluate.add_argument("--method", choices=("dexpilot",))
    evaluate.add_argument("--run-id")
    evaluate.add_argument("--device")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--scaling-factor", type=float, default=1.0)
    evaluate.add_argument("--low-pass-alpha", type=float, default=0.2)
    return parser


def _capture_metadata(path, mocap):
    config = load_json(path)
    if config.get("mocap", mocap) != mocap:
        raise ValueError("capture config mocap does not match --mocap")
    hand_side = config.get("hand_side")
    if hand_side not in {"left", "right"}:
        raise ValueError("capture config hand_side must be 'left' or 'right'")
    calibration = config.get("calibration", {})
    scale, rotation, outward_sign = validate_calibration(
        calibration.get("scale"),
        calibration.get("rotation"),
        calibration.get("outward_sign"),
    )
    return {
        "mocap": mocap,
        "hand_side": hand_side,
        "calibration": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "outward_sign": outward_sign,
        },
    }


def _load_observations(input_path, timestamps_path):
    loaded = np.load(input_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            observations = loaded["observations"]
            embedded_timestamps = loaded["timestamps"] if "timestamps" in loaded else None
        finally:
            loaded.close()
    else:
        observations = loaded
        embedded_timestamps = None
    if observations.ndim < 1 or not len(observations):
        raise ValueError("input must contain observations")
    timestamps = (
        np.load(timestamps_path, allow_pickle=False)
        if timestamps_path
        else embedded_timestamps
    )
    if timestamps is None:
        timestamps = np.arange(len(observations), dtype=np.float64) / 30
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if (
        timestamps.shape != (len(observations),)
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) < 0)
    ):
        raise ValueError("timestamps must be finite, monotonic, and match observations")
    return observations, timestamps


def _collect(args):
    metadata = _capture_metadata(args.config, args.mocap)
    if not args.input:
        from geort.rollout import collect_live

        observations, timestamps = collect_live(args.mocap, args.config)
    else:
        observations, timestamps = _load_observations(
            args.input, args.timestamps
        )
    frames = [
        adapt_observation(
            args.mocap,
            observation,
            timestamp,
            metadata["hand_side"],
            metadata["calibration"],
        )
        for observation, timestamp in zip(observations, timestamps)
    ]
    dataset = create_dataset(args.dataset, observations, frames, metadata)
    print(dataset)
    return dataset


def _run_config(dataset, robot_name, robot, method, seed, options=None):
    robot_path = Path(robot_name)
    if robot_path.is_absolute() or ".." in robot_path.parts:
        raise ValueError("robot config path must be project-relative")
    return {
        "dataset": dataset,
        "robot": robot_name,
        "robot_fingerprint": robot["robot_fingerprint"],
        "method": method,
        "seed": seed,
        "hand_side": robot["hand_side"],
        "method_options": options or {},
    }


def _train(args):
    if args.resume:
        if args.dataset or args.robot or args.run_id:
            raise ValueError("--resume cannot be combined with --dataset, --robot, or --run-id")
        if args.epoch is None:
            raise ValueError("resume requires --epoch as the total epoch count")
        run, config = load_run(args.resume)
        if config.get("method") != "geort":
            raise ValueError("only GeoRT runs can be resumed")
        robot = _load_robot_for_run(config)
        dataset_id = config["dataset"]
        saved = config.get("training", {})
        missing = set(_RESUME_SETTINGS) - saved.keys()
        if missing:
            raise ValueError(f"run training config is missing: {sorted(missing)}")
        train_options = {name: saved[name] for name in _RESUME_SETTINGS}
        for name in _TRAIN_DEFAULTS:
            value = getattr(args, name)
            if value is not None:
                train_options[name] = value
        train_options.update(epoch=args.epoch, resume=run / "checkpoints")
    else:
        if not args.dataset or not args.robot:
            raise ValueError("new training requires --dataset and --robot")
        dataset_id = args.dataset
        robot = load_robot(args.robot)
        train_options = {
            name: default if getattr(args, name) is None else getattr(args, name)
            for name, default in _TRAIN_DEFAULTS.items()
        }
        train_options["epoch"] = 50 if args.epoch is None else args.epoch
        run_id = args.run_id or (
            f"{dataset_id}_{args.robot}_geort_seed{train_options['seed']}"
        )
        run = create_run(
            run_id,
            _run_config(
                dataset_id,
                args.robot,
                robot,
                "geort",
                train_options["seed"],
            ),
        )

    _, frames, metadata = load_dataset(dataset_id)
    if metadata["hand_side"] != robot["hand_side"]:
        raise ValueError("dataset hand_side does not match robot")
    valid_points = [frame.points for frame in frames if frame.valid]
    if len(valid_points) < 2:
        raise ValueError("training requires at least two valid frames")
    points = np.stack(valid_points)
    from geort.trainer import GeoRTTrainer

    trainer = GeoRTTrainer(
        robot,
        device=args.device,
        checkpoint_dir=run / "checkpoints",
    )
    trainer.train(points, **train_options)
    print(run)
    return run


def _load_robot_for_run(config):
    robot = load_robot(config["robot"])
    if config.get("robot_fingerprint") != robot["robot_fingerprint"]:
        raise ValueError("robot configuration or assets changed after run creation")
    return robot


def _build_hand(robot):
    from geort.env.hand import HandKinematicModel

    return HandKinematicModel.build_from_config(robot)


def _evaluate(args):
    if args.run:
        if args.dataset or args.robot or args.method or args.run_id:
            raise ValueError("--run cannot be combined with a new evaluation run")
        run, config = load_run(args.run)
        robot = _load_robot_for_run(config)
    else:
        if not (args.dataset and args.robot and args.method == "dexpilot"):
            raise ValueError(
                "use --run, or --dataset/--robot/--method dexpilot for a new baseline run"
            )
        robot = load_robot(args.robot)
        run_id = args.run_id or (
            f"{args.dataset}_{args.robot}_dexpilot_seed{args.seed}"
        )
        config = _run_config(
            args.dataset,
            args.robot,
            robot,
            "dexpilot",
            args.seed,
            {
                "scaling_factor": args.scaling_factor,
                "low_pass_alpha": args.low_pass_alpha,
            },
        )

    _, all_frames, metadata = load_dataset(config["dataset"])
    if metadata["hand_side"] != robot["hand_side"]:
        raise ValueError("dataset hand_side does not match robot")
    if not args.run:
        run = create_run(run_id, config)
    frames = [frame for frame in all_frames if frame.valid]
    if not frames:
        raise ValueError("evaluation requires at least one valid frame")
    method = make_method(config, robot, run, device=args.device)
    commands = []
    latencies = []
    for frame in frames:
        start = time.perf_counter()
        commands.append(method.infer(frame))
        latencies.append(time.perf_counter() - start)
    save_commands(run, commands, overwrite=True)
    latencies = np.asarray(latencies, dtype=np.float64)
    np.save(run / "outputs" / "latency.npy", latencies)
    metrics = evaluate_trajectory(
        frames,
        np.stack([command.qpos for command in commands]),
        commands[0].joint_names,
        latencies,
        robot,
        _build_hand(robot),
    )
    save_json(metrics, run / "outputs" / "metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def _rollout(args):
    if args.source == "live" and not args.config:
        raise ValueError("live rollout requires --config")
    from geort.rollout import run

    return run(args.run, args.source, args.device, args.config)


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {
        "collect": _collect,
        "train": _train,
        "rollout": _rollout,
        "evaluate": _evaluate,
    }[args.command](args)
