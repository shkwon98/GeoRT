import numpy as np
import pytest

import geort
import geort.utils.path as path_utils
from geort.utils.config_utils import get_config, save_json
from geort.utils.path import get_human_data


def test_config_and_human_data_lookup_are_exact(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.json"
    data_path = tmp_path / "human.npy"
    save_json({"source": "explicit"}, config_path)
    np.save(data_path, np.zeros((1, 1, 3)))
    monkeypatch.setattr(path_utils, "get_data_root", lambda: tmp_path)

    assert get_config("allegro_right")["joint_order"]
    assert get_config(config_path) == {"source": "explicit"}
    assert get_human_data("human.npy") == data_path
    with pytest.raises(FileNotFoundError):
        get_config("allegro")


def test_save_human_data_returns_the_existing_npy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GEORT_HOME", str(tmp_path))
    points = np.zeros((2, 21, 3), dtype=np.float32)

    saved = geort.save_human_data(points, "recording")

    assert saved == tmp_path / "data" / "recording.npy"
    assert saved.is_file()
    np.testing.assert_array_equal(np.load(saved), points)
    with pytest.raises(ValueError, match=".npy"):
        geort.save_human_data(points, "recording.csv")
