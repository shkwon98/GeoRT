import numpy as np

from experiments.schema import NamedCommand


class DexPilotMethod:
    def __init__(self, robot_spec, options=None, retargeter=None):
        options = options or {}
        unknown = set(options) - {"scaling_factor", "low_pass_alpha"}
        if unknown:
            raise ValueError(f"unknown DexPilot options: {sorted(unknown)}")

        self.joint_names = tuple(robot_spec["joint_order"])
        if retargeter is None:
            from dex_retargeting.retargeting_config import RetargetingConfig

            finger_order = ("thumb", "index", "middle", "ring", "pinky")
            tips_by_name = {
                tip["name"]: tip["link"] for tip in robot_spec["fingertip_link"]
            }
            tip_names = [
                tips_by_name[name] for name in finger_order if name in tips_by_name
            ]
            config = RetargetingConfig.from_dict(
                {
                    "type": "dexpilot",
                    "urdf_path": robot_spec["urdf_path"],
                    "target_joint_names": list(self.joint_names),
                    "wrist_link_name": robot_spec["base_link"],
                    "finger_tip_link_names": tip_names,
                    "scaling_factor": float(options.get("scaling_factor", 1.0)),
                    "low_pass_alpha": float(options.get("low_pass_alpha", 0.2)),
                    "has_joint_limits": True,
                }
            )
            retargeter = config.build()
        self.retargeter = retargeter

        source_names = tuple(retargeter.joint_names)
        if (
            not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
            or len(set(source_names)) != len(source_names)
            or set(source_names) != set(self.joint_names)
        ):
            raise ValueError("DexPilot and robot joint names must match uniquely")
        self._source_indices = [source_names.index(name) for name in self.joint_names]

    def infer(self, frame):
        if not frame.valid:
            raise ValueError("cannot infer from an invalid canonical frame")
        indices = np.asarray(
            self.retargeter.optimizer.target_link_human_indices, dtype=np.int64
        )
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError("DexPilot human landmark indices must have shape (2, N)")
        reference = frame.points[indices[1]] - frame.points[indices[0]]
        source_qpos = np.asarray(self.retargeter.retarget(reference))
        return NamedCommand(
            self.joint_names,
            source_qpos[self._source_indices],
            frame.timestamp,
        )
