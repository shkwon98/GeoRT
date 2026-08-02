# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

# GeoRT Package Path Helper.
# No more pain configuring paths!
def get_package_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve()

def to_package_root(path):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve() / path

def get_data_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "data").resolve()

def get_checkpoint_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "checkpoint").resolve()

def get_human_data_output_path(human_data):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "data" / human_data).resolve()

def get_human_data(name_or_path):
    path = Path(name_or_path)
    if path.is_file():
        return path

    path = Path(get_data_root()) / path
    if not path.suffix:
        path = path.with_suffix(".npy")
    if path.is_file():
        return path
    raise FileNotFoundError(f"Human data file not found: {path}")


if __name__ == '__main__':
    print(get_package_root())
    print(get_human_data_output_path("human"))
