import hashlib
import json
import os
from pathlib import Path
from string import Template

from geort.utils.config_utils import get_config
from geort.utils.path import resolve_resource_path


_ROOT = Path(__file__).parent
_SPECS = {
    "allegro_left": _ROOT / "allegro" / "left.json",
    "allegro_right": _ROOT / "allegro" / "right.json",
    "wuji_left": _ROOT / "wuji" / "left.json",
    "wuji_right": _ROOT / "wuji" / "right.json",
}


def load_robot_spec(name, env=None):
    try:
        spec_path = _SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown robot: {name}") from error
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    if "geort_config" in spec:
        config = get_config(spec["geort_config"])
        resolved = {**config, **spec}
        urdf_path = resolve_resource_path(resolved["urdf_path"])
    else:
        environment = os.environ if env is None else env
        try:
            expanded = Template(spec["urdf_path"]).substitute(environment)
        except KeyError as error:
            raise ValueError(
                f"missing environment variable {error.args[0]} for {name}"
            ) from error
        urdf_path = Path(expanded).expanduser()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"robot URDF not found: {urdf_path}")
        resolved = spec

    joint_order = resolved.get("joint_order", [])
    tips = resolved.get("fingertip_link", [])
    grouped_joints = [joint for tip in tips for joint in tip.get("joint", [])]
    if (
        resolved.get("name") != name
        or resolved.get("hand_side") not in {"left", "right"}
        or not joint_order
        or len(set(joint_order)) != len(joint_order)
        or len(grouped_joints) != len(joint_order)
        or set(grouped_joints) != set(joint_order)
    ):
        raise ValueError(f"invalid robot specification: {name}")

    resolved = dict(resolved)
    resolved["urdf_path"] = str(urdf_path.resolve())
    resolved["urdf_sha256"] = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    return resolved
