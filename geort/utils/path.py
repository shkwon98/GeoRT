# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path


def get_package_root():
    return Path(__file__).resolve().parents[1]


def get_resource_root():
    return get_package_root() / "resources"


def _resource_relative_path(path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Expected a package-relative resource path, got: {path}")
    return Path(*(part for part in path.parts if part != "."))


def get_resource_path(path):
    resource_path = get_resource_root() / _resource_relative_path(path)
    if not resource_path.exists():
        raise FileNotFoundError(
            f"Packaged resource not found: {resource_path}")
    return resource_path


def resolve_resource_path(path):
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        raise FileNotFoundError(f"Resource file not found: {path}")
    resource_path = get_resource_root() / _resource_relative_path(path)
    if resource_path.is_file():
        return resource_path
    raise FileNotFoundError(f"Resource file not found: {path}")


def get_hand_landmarker_path():
    return get_resource_path("hand_landmarker.task")


def get_bundled_fk_checkpoint(name):
    return get_resource_path(Path("models") / f"fk_model_{name}.pth")


def _get_writable_root():
    configured_root = os.environ.get("GEORT_HOME")
    if configured_root:
        return Path(configured_root).expanduser()

    source_root = get_package_root().parent
    if (source_root / ".git").exists():
        return source_root / ".geort"

    return Path.cwd() / ".geort"


def get_data_root():
    return _get_writable_root() / "data"


def get_run_root():
    return _get_writable_root() / "runs"


def get_robot_cache_root(robot):
    name = robot.get("name")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("robot name must be a single path component")
    fingerprint = robot.get("urdf_sha256")
    if fingerprint:
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) < 12
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint.lower()
            )
        ):
            raise ValueError("urdf_sha256 must be a hexadecimal digest")
        name = f"{name}-{fingerprint[:12].lower()}"
    return _get_writable_root() / "cache" / name


def get_checkpoint_root():
    return get_run_root()


def get_human_data_output_path(human_data):
    output_path = get_data_root() / human_data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def get_human_data(name_or_path):
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return path
    if path.is_absolute():
        raise FileNotFoundError(f"Human data file not found: {path}")

    if not path.suffix:
        path = path.with_suffix(".npy")
    data_path = get_data_root() / path
    if data_path.is_file():
        return data_path

    if ".." not in path.parts:
        bundled_path = get_resource_root() / "samples" / path
        if bundled_path.is_file():
            return bundled_path
    raise FileNotFoundError(f"Human data file not found: {data_path}")
