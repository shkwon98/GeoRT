import numpy as np
import pytest

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
