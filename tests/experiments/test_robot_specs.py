from pathlib import Path

import pytest

from experiments.robots import load_robot_spec


def test_allegro_and_wuji_specs_have_complete_named_joint_groups(tmp_path):
    description = tmp_path / "wuji_hand_description"
    (description / "urdf").mkdir(parents=True)
    (description / "urdf" / "right.urdf").write_text("<robot name='right'/>")
    wuji = load_robot_spec(
        "wuji_right", env={"WUJI_HAND_DESCRIPTION": str(description)}
    )
    allegro = load_robot_spec("allegro_right")

    assert Path(wuji["urdf_path"]).is_file()
    assert len(wuji["urdf_sha256"]) == 64
    assert len(wuji["joint_order"]) == 20
    assert len(
        {name for tip in wuji["fingertip_link"] for name in tip["joint"]}
    ) == 20
    assert wuji["ros"]["input_topic"] == "/teleop/human/hand_right/pose"
    assert wuji["ros"]["output_topic"] == (
        "/control/hand_right/hand_right_controller/joint_trajectory"
    )
    assert allegro["geort_config"] == "allegro_right"


def test_wuji_spec_requires_the_external_description_path():
    with pytest.raises(ValueError, match="WUJI_HAND_DESCRIPTION"):
        load_robot_spec("wuji_left", env={})
