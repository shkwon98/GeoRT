from types import SimpleNamespace

import numpy as np

from geort.mocap import mediapipe_mocap


def test_mediapipe_mocap_returns_metric_world_landmarks(monkeypatch):
    world = np.arange(63, dtype=np.float32).reshape(21, 3)
    mocap = mediapipe_mocap.MediaPipeMocap.__new__(
        mediapipe_mocap.MediaPipeMocap
    )
    mocap.camera = SimpleNamespace(
        get_frame=lambda: {"rgb": np.zeros((2, 2, 3)), "depth": None}
    )
    mocap.detector = SimpleNamespace(
        detect=lambda image, hand_side: (
            {
                "detected": hand_side == "left",
                "annotated_img": image,
                "canonical_coordinates": np.zeros((21, 3)),
                "world_coordinates": world,
            }
        )
    )
    mocap.status = "idle"
    mocap.hand_side = "left"
    mocap.cv2 = SimpleNamespace(
        imshow=lambda *args: None,
        waitKey=lambda delay: ord("s"),
    )

    result = mocap.get()

    assert result["status"] == "recording"
    assert result["result"] is world
