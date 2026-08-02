import numpy as np
import pytest

from geort.utils.config_utils import get_config, save_json
from geort.utils.path import get_human_data


def test_get_config_uses_exact_name_or_existing_path(tmp_path):
    explicit_config = tmp_path / "custom.json"
    save_json({"source": "explicit"}, explicit_config)

    assert get_config("allegro_right")["joint_order"]
    assert get_config(explicit_config) == {"source": "explicit"}

    with pytest.raises(FileNotFoundError, match="allegro.json"):
        get_config("allegro")


def test_get_human_data_uses_exact_name_or_existing_path(tmp_path):
    explicit_data = tmp_path / "custom.npy"
    np.save(explicit_data, np.array([[1.0, 2.0, 3.0]]))

    assert get_human_data("human_alex").name == "human_alex.npy"
    assert get_human_data(explicit_data) == explicit_data

    with pytest.raises(FileNotFoundError, match="human.npy"):
        get_human_data("human")
