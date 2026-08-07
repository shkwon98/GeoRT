import pytest

import geort.utils.path as path_utils
from geort.utils.config_utils import get_config, save_json


def test_source_checkout_uses_project_local_geort_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    package_root = project_root / "geort"
    package_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    monkeypatch.delenv("GEORT_HOME", raising=False)
    monkeypatch.setattr(path_utils, "get_package_root", lambda: package_root)

    assert path_utils.get_data_root() == project_root / ".geort" / "data"
    assert path_utils.get_run_root() == project_root / ".geort" / "runs"


def test_robot_cache_is_partitioned_by_robot_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("GEORT_HOME", str(tmp_path / ".geort"))
    robot = {"name": "wuji_right", "robot_fingerprint": "0123456789abcdef"}

    get_robot_cache_root = getattr(
        path_utils, "get_robot_cache_root", lambda unused: None
    )

    assert get_robot_cache_root(robot) == (
        tmp_path / ".geort" / "cache" / "wuji_right-0123456789ab"
    )
    assert get_robot_cache_root({"name": "allegro_right"}) == (
        tmp_path / ".geort" / "cache" / "allegro_right"
    )


def test_config_lookup_is_exact(tmp_path):
    config_path = tmp_path / "custom.json"
    save_json({"source": "explicit"}, config_path)

    assert get_config("allegro_right")["joint_order"]
    assert get_config(config_path) == {"source": "explicit"}
    with pytest.raises(FileNotFoundError):
        get_config("allegro")
