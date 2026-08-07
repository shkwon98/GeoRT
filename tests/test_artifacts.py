import json

import numpy as np


def test_dataset_and_run_artifacts_have_separate_roots(tmp_path):
    from geort.artifacts import (
        create_dataset,
        create_run,
        load_dataset,
        load_run,
        save_commands,
    )
    from geort.schema import CanonicalFrame, NamedCommand

    observations = np.zeros((2, 21, 3), dtype=np.float32)
    frames = [
        CanonicalFrame(np.zeros((21, 3)), timestamp, "right")
        for timestamp in (1.0, 1.1)
    ]
    dataset = create_dataset(
        "manus_right_001",
        observations,
        frames,
        {
            "mocap": "manus",
            "hand_side": "right",
            "calibration": {
                "scale": 1.0,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "outward_sign": 1,
            },
        },
        root=tmp_path,
    )
    run = create_run(
        "manus_wuji_geort_seed0",
        {
            "dataset": "manus_right_001",
            "robot": "wuji_right",
            "method": "geort",
            "seed": 0,
        },
        root=tmp_path,
    )
    command_path = save_commands(
        run,
        [NamedCommand(("joint_0",), np.array([0.2]), 1.0)],
    )

    assert dataset == tmp_path / "data" / "manus_right_001"
    assert {path.name for path in dataset.iterdir()} == {
        "raw.npz",
        "canonical.npy",
        "metadata.json",
    }
    assert json.loads((dataset / "metadata.json").read_text())["valid"] == [
        True,
        True,
    ]
    assert run == tmp_path / "runs" / "manus_wuji_geort_seed0"
    assert (run / "checkpoints").is_dir()
    assert command_path == run / "outputs" / "qpos.npz"
    assert not (run / "raw.npz").exists()

    loaded_observations, loaded_frames, loaded_metadata = load_dataset(
        "manus_right_001", root=tmp_path
    )
    loaded_run, loaded_config = load_run(
        "manus_wuji_geort_seed0", root=tmp_path
    )
    np.testing.assert_array_equal(loaded_observations, observations)
    np.testing.assert_array_equal(loaded_frames[0].points, frames[0].points)
    assert loaded_metadata["mocap"] == "manus"
    assert loaded_run == run
    assert loaded_config["robot"] == "wuji_right"
