from geort.utils.config_utils import get_config
from geort.utils.path import (
    get_bundled_fk_checkpoint,
    get_checkpoint_root,
    get_data_root,
    get_hand_landmarker_path,
    get_human_data,
    get_human_data_output_path,
    resolve_resource_path,
)


def test_packaged_resources_and_writable_root(monkeypatch, tmp_path):
    config = get_config("allegro_right")
    monkeypatch.setenv("GEORT_HOME", str(tmp_path))

    assert resolve_resource_path(config["urdf_path"]).is_file()
    assert get_hand_landmarker_path().is_file()
    assert get_human_data("human_alex").is_file()
    assert get_bundled_fk_checkpoint("allegro_right").is_file()
    assert get_data_root() == tmp_path / "data"
    assert get_checkpoint_root() == tmp_path / "runs"
    assert get_human_data_output_path(
        "recording") == tmp_path / "data" / "recording"
