from pathlib import Path

from experiments.schema import NamedCommand
from geort.export import load_model


_OPTIONS = {
    "device",
    "tag",
    "epoch",
    "seed",
    "val_fraction",
    "coverage_samples",
    "coverage_batch_size",
    "gesture_batch_size",
    "direction_sigma",
    "flatness_sigma",
    "w_chamfer",
    "w_curvature",
    "w_pinch",
    "w_collision",
}


class GeoRTMethod:
    def __init__(self, checkpoint_dir, robot_spec, device=None):
        self.model = load_model(checkpoint_dir, device=device)
        self.joint_names = tuple(robot_spec["joint_order"])
        limits = self.model.qpos_normalizer.joint_lower_limit
        if len(self.joint_names) != len(limits):
            raise ValueError("robot joint_order does not match the GeoRT checkpoint")

    def infer(self, frame):
        if not frame.valid:
            raise ValueError("cannot infer from an invalid canonical frame")
        return NamedCommand(
            self.joint_names,
            self.model.forward(frame.points),
            frame.timestamp,
        )


def train(canonical_path, robot_spec, run_dir, options):
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown GeoRT options: {sorted(unknown)}")

    from geort.trainer import GeoRTTrainer

    run_dir = Path(run_dir)
    trainer = GeoRTTrainer(
        robot_spec,
        device=options.get("device"),
        data_dir=run_dir / "robot_data",
        checkpoint_dir=run_dir / "checkpoints",
    )
    train_options = {
        "tag": options.get("tag", "geort"),
        "epoch": int(options.get("epoch", 50)),
        "seed": int(options.get("seed", 0)),
        "val_fraction": float(options.get("val_fraction", 0.1)),
        "coverage_samples": int(options.get("coverage_samples", 20000)),
        "coverage_batch_size": int(options.get("coverage_batch_size", 2048)),
        "gesture_batch_size": int(options.get("gesture_batch_size", 2048)),
    }
    for name in (
        "direction_sigma",
        "flatness_sigma",
        "w_chamfer",
        "w_curvature",
        "w_pinch",
        "w_collision",
    ):
        if name in options:
            train_options[name] = float(options[name])
    return trainer.train(canonical_path, **train_options)
