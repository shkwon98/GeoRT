import importlib
import sys
import types

import pytest


def _load_mediapipe_mocap(monkeypatch, key):
    mediapipe = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    framework = types.ModuleType("mediapipe.framework")
    formats = types.ModuleType("mediapipe.framework.formats")
    landmark_pb2 = types.ModuleType("mediapipe.framework.formats.landmark_pb2")
    tasks = types.ModuleType("mediapipe.tasks")
    python = types.ModuleType("mediapipe.tasks.python")
    vision = types.ModuleType("mediapipe.tasks.python.vision")
    cv2 = types.ModuleType("cv2")
    cv2.imshow = lambda *_: None
    cv2.waitKey = lambda _: key
    realsense = types.ModuleType("geort.mocap.camera.realsense")
    realsense.RealSenseCamera = object

    mediapipe.solutions = solutions
    mediapipe.framework = framework
    framework.formats = formats
    formats.landmark_pb2 = landmark_pb2
    mediapipe.tasks = tasks
    tasks.python = python
    python.vision = vision

    for name, module in {
        "mediapipe": mediapipe,
        "mediapipe.solutions": solutions,
        "mediapipe.framework": framework,
        "mediapipe.framework.formats": formats,
        "mediapipe.framework.formats.landmark_pb2": landmark_pb2,
        "mediapipe.tasks": tasks,
        "mediapipe.tasks.python": python,
        "mediapipe.tasks.python.vision": vision,
        "pyrealsense2": types.ModuleType("pyrealsense2"),
        "cv2": cv2,
        "geort.mocap.camera.realsense": realsense,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "geort.mocap.mediapipe_mocap", raising=False)
    return importlib.import_module("geort.mocap.mediapipe_mocap")


@pytest.mark.parametrize(
    ("key", "initial_status", "expected_status"),
    [
        (ord("q"), "idle", "quit"),
        (ord("s"), "idle", "recording"),
        (ord("e"), "recording", "idle"),
    ],
)
def test_mediapipe_mocap_keyboard_controls_status(monkeypatch, key, initial_status, expected_status):
    module = _load_mediapipe_mocap(monkeypatch, key)
    mocap = module.MediaPipeMocap.__new__(module.MediaPipeMocap)
    mocap.status = initial_status
    mocap.camera = type("Camera", (), {"get_frame": lambda self: {"rgb": object()}})()
    mocap.detector = type(
        "Detector",
        (),
        {"detect": lambda self, _: {"annotated_img": object(), "detected": False}},
    )()

    assert mocap.get() == {"status": expected_status, "result": None}
