import json

import numpy as np
import pytest

from experiments.artifacts import (
    create_run_dir,
    save_canonical_recording,
    save_commands,
    save_raw_recording,
)
from experiments.schema import CanonicalFrame, NamedCommand


def test_artifacts_keep_raw_and_derived_data_separate(tmp_path):
    run = create_run_dir("run-a", root=tmp_path)
    raw = np.arange(2 * 25 * 3, dtype=np.float32).reshape(2, 25, 3)
    timestamps = np.array([1.0, 1.1])
    raw_path = save_raw_recording(
        run,
        raw,
        timestamps,
        {
            "mocap": "webxr",
            "hand_side": "right",
            "units": "m",
            "source_frame": "webxr",
            "calibration": {"scale": 1.0},
        },
    )
    frames = [
        CanonicalFrame(np.zeros((21, 3)), timestamp, "right")
        for timestamp in timestamps
    ]
    canonical_path = save_canonical_recording(run, frames, {"mocap": "webxr"})
    command_path = save_commands(
        run, [NamedCommand(("j0",), np.array([0.2]), 1.0)]
    )

    assert raw_path.name == "raw.npz"
    assert canonical_path.name == "canonical.npy"
    assert command_path.name == "qpos.npz"
    np.testing.assert_array_equal(np.load(raw_path)["observations"], raw)
    assert json.loads((run / "raw_metadata.json").read_text())["mocap"] == "webxr"
    with pytest.raises(FileExistsError):
        save_raw_recording(run, raw, timestamps, {"mocap": "webxr"})
