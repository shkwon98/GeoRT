# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).parent / "geort"
RESOURCE_FILES = [
    path.relative_to(PACKAGE_ROOT).as_posix()
    for path in (PACKAGE_ROOT / "resources").rglob("*")
    if path.is_file()
]


setup(
    name="geort",
    version="0.1",
    packages=find_packages(include=["geort", "geort.*"]),
    include_package_data=True,
    package_data={"geort": ["config/*.json", *RESOURCE_FILES]},
    python_requires=">=3.8",
    install_requires=["numpy>=1.23,<2", "torch>=2.2,<2.4"],
    extras_require={
        "training": ["tqdm>=4.66,<5", "open3d>=0.13,<0.19", "sapien>=2.2,<3"],
        "mediapipe": [
            "mediapipe==0.10.9; python_version < '3.9'",
            "mediapipe>=0.10.14,<0.11; python_version >= '3.9'",
            "opencv-python==4.8.1.78; python_version < '3.9'",
            "opencv-contrib-python==4.8.1.78; python_version < '3.9'",
            "opencv-python>=4.8,<5; python_version >= '3.9'",
            "pyrealsense2>=2.54,<3",
            "scipy>=1.10,<2",
        ],
        "dev": ["build>=1,<2", "pytest>=7,<9"],
    },
)
