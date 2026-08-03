import importlib
import sys
from types import SimpleNamespace

import pytest


class Pipeline:
    def __init__(self, result):
        self.result = result

    def wait_for_frames(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def stop(self):
        pass


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(
            get_color_frame=lambda: None,
            get_depth_frame=lambda: None,
        ),
        RuntimeError("camera"),
    ],
)
def test_realsense_no_frame_paths_share_one_result(monkeypatch, result):
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    sys.modules.pop("geort.mocap.camera.realsense", None)
    module = importlib.import_module("geort.mocap.camera.realsense")
    camera = module.RealSenseCamera.__new__(module.RealSenseCamera)
    camera.pipeline = Pipeline(result)

    assert camera.get_frame() == {"rgb": None, "depth": None}
