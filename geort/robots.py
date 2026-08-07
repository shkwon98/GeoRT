import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from geort.utils.config_utils import get_config
from geort.utils.path import resolve_resource_path


def robot_fingerprint(config, urdf_path):
    urdf_path = Path(urdf_path)
    fingerprint_config = {
        key: value for key, value in config.items() if key != "urdf_path"
    }
    digest = hashlib.sha256(
        json.dumps(
            fingerprint_config, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    digest.update(urdf_path.read_bytes())
    filenames = {
        mesh.attrib["filename"]
        for mesh in ET.parse(urdf_path).iter("mesh")
        if "filename" in mesh.attrib
    }
    for filename in sorted(filenames):
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"robot mesh not found: {filename}")
        digest.update(filename.encode())
        digest.update(mesh_path.read_bytes())
    return digest.hexdigest()


def load_robot(name_or_path):
    config_path = Path(name_or_path)
    robot = get_config(name_or_path)
    urdf = Path(robot.get("urdf_path", ""))
    if config_path.is_file() and not urdf.is_absolute():
        candidate = config_path.resolve().parent / urdf
        urdf = candidate if candidate.is_file() else resolve_resource_path(urdf)
    else:
        urdf = resolve_resource_path(urdf)

    joint_order = robot.get("joint_order", [])
    grouped = [
        joint
        for fingertip in robot.get("fingertip_link", [])
        for joint in fingertip.get("joint", [])
    ]
    if (
        robot.get("hand_side") not in {"left", "right"}
        or not joint_order
        or len(set(joint_order)) != len(joint_order)
        or len(grouped) != len(joint_order)
        or set(grouped) != set(joint_order)
    ):
        raise ValueError(f"invalid robot configuration: {name_or_path}")

    resolved = dict(robot)
    resolved["urdf_path"] = str(urdf.resolve())
    resolved["robot_fingerprint"] = robot_fingerprint(robot, urdf)
    return resolved
