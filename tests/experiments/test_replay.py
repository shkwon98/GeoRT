import json

import numpy as np

from experiments.mocap.replay import load_replay


def test_replay_preserves_timestamps_side_and_validity(tmp_path):
    points = np.zeros((2, 21, 3), dtype=np.float32)
    np.save(tmp_path / "canonical.npy", points)
    (tmp_path / "canonical_metadata.json").write_text(
        json.dumps(
            {
                "hand_side": "left",
                "timestamps": [4.0, 4.1],
                "valid": [True, False],
            }
        )
    )

    frames = load_replay(
        tmp_path / "canonical.npy", tmp_path / "canonical_metadata.json"
    )

    assert [frame.timestamp for frame in frames] == [4.0, 4.1]
    assert [frame.valid for frame in frames] == [True, False]
    assert all(frame.hand_side == "left" for frame in frames)
